import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm, trange
from trainer.lamb import Lamb
from retrieve_utils import load_index
from utils import set_seed, metric_weights, save_model
from torch.utils.data import RandomSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


def adoretrain(args, model, train_dataset, collate_fn, passage_embeddings, tb_writer, logger):
    """ Train the model """

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer_str == 'adamw':
        optimizer = optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon
        )
    elif args.optimizer_str == 'lamb':
        optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise NotImplementedError("Optimizer must be adamw or lamb")

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    args.faiss_gpus = [i for i in range(args.n_gpu)]

    index = load_index(passage_embeddings, args.index_path, args.faiss_gpus, not args.index_cpu)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_mrr, logging_mrr = 0.0, 0.0
    tr_recall, logging_recall = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(seed=args.seed, n_gpu=args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, _, all_rel_poffsets) in enumerate(epoch_iterator):

            batch = {k: v.to(args.device) for k, v in batch.items()}
            model.train()
            query_embeddings = model(
                query_ids=batch["input_ids"],
                attention_mask_q=batch["attention_mask"],
                is_query=True)
            I_nearest_neighbor = index.search(
                query_embeddings.detach().cpu().numpy(), args.neg_topk)[1]

            loss = 0
            for retrieve_poffsets, cur_rel_poffsets, qembedding in zip(
                    I_nearest_neighbor, all_rel_poffsets, query_embeddings):
                target_labels = np.isin(retrieve_poffsets, cur_rel_poffsets).astype(np.int32)

                # get target_labels index
                first_rel_pos = np.where(target_labels[:10])[0]
                mrr = 1 / (1 + first_rel_pos[0]) if len(first_rel_pos) > 0 else 0

                tr_mrr += mrr / args.train_batch_size
                recall = 1 if mrr > 0 else 0
                tr_recall += recall / args.train_batch_size

                if np.sum(target_labels) == 0:
                    retrieve_poffsets = np.hstack([retrieve_poffsets, cur_rel_poffsets])
                    target_labels = np.hstack([target_labels, [True] * len(cur_rel_poffsets)])

                target_labels = target_labels.reshape(-1, 1)
                rel_diff = target_labels - target_labels.T
                pos_pairs = (rel_diff > 0).astype(np.float32)
                num_pos_pairs = np.sum(pos_pairs, (0, 1))

                assert num_pos_pairs > 0
                neg_pairs = (rel_diff < 0).astype(np.float32)
                num_pairs = 2 * num_pos_pairs  # num pos pairs and neg pairs are always the same

                pos_pairs = torch.FloatTensor(pos_pairs).to(args.device)
                neg_pairs = torch.FloatTensor(neg_pairs).to(args.device)

                topK_passage_embeddings = torch.FloatTensor(
                    passage_embeddings[retrieve_poffsets]).to(args.model_device)
                y_pred = (qembedding.unsqueeze(0) * topK_passage_embeddings).sum(-1, keepdim=True)

                C_pos = torch.log(1 + torch.exp(y_pred - y_pred.t()))
                C_neg = torch.log(1 + torch.exp(y_pred - y_pred.t()))

                C = pos_pairs * C_pos + neg_pairs * C_neg

                if args.metric_cut is not None:
                    with torch.no_grad():
                        weights = metric_weights(y_pred, args.metric_cut)
                    C = C * weights
                cur_loss = torch.sum(C, (0, 1)) / num_pairs
                loss += cur_loss

            loss /= (args.train_batch_size * args.gradient_accumulation_steps)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    cur_loss = (tr_loss - logging_loss) / args.logging_steps
                    tb_writer.add_scalar('train/all_loss', cur_loss, global_step)
                    logging_loss = tr_loss

                    cur_mrr = (tr_mrr - logging_mrr) / (
                            args.logging_steps * args.gradient_accumulation_steps)
                    tb_writer.add_scalar('train/mrr_10', cur_mrr, global_step)
                    logging_mrr = tr_mrr

                    cur_recall = (tr_recall - logging_recall) / (
                            args.logging_steps * args.gradient_accumulation_steps)
                    tb_writer.add_scalar('train/recall_10', cur_recall, global_step)
                    logging_recall = tr_recall

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(model, args.model_save_dir, 'ckpt-{}'.format(global_step), args)

        save_model(model, args.model_save_dir, 'epoch-{}'.format(epoch_idx + 1), args)
