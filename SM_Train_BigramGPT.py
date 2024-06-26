import argparse
import json
import logging
import os
import sys
from pathlib import Path

#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaTokenizer

from BigramGPT import BigramLanguageModel
from dataloader import build_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info("Loaded all moduels")

def train(args):
    logger.info("Started training job")
    # is_distributed = len(args.hosts) > 1 and args.backend is not None
    # logger.debug("Distributed training - {}".format(is_distributed))
    # use_cuda = args.num_gpus > 0  # two diff settings, not good
    # logger.debug("Number of gpus available - {}".format(args.num_gpus))
    # kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    # device = torch.device("cuda" if use_cuda else "cpu")

    # if is_distributed:
    #     # Initialize the distributed environment.
    #     world_size = len(args.hosts)
    #     os.environ["WORLD_SIZE"] = str(world_size)
    #     host_rank = args.hosts.index(args.current_host)
    #     os.environ["RANK"] = str(host_rank)
    #     dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
    #     logger.info(
    #         "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
    #             args.backend, dist.get_world_size()
    #         )
    #         + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
    #     )

    # # set the seed for generating random numbers
    # torch.manual_seed(args.seed)
    # if use_cuda:
    #     torch.cuda.manual_seed(args.seed)

    # add parameters
    merges_file, vocab_file = getEncoderFiles(args)
    logger.info(f"Found log files {merges_file} and {vocab_file}.")
    #tokenizer = RobertaTokenizer.from_pretrained(args.vocab_file, args.merges_file, max_length=512)
    tokenizer = RobertaTokenizer(vocab_file=vocab_file, merges_file=merges_file, max_length=512)
    # BPE tokenizer that comes with preprocessing and in a compatible format

    collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # parameter
    if args.data_dir[-4:] != ".txt":
        datafiles = [f"{args.data_dir}/{f}" for f in os.listdir(args.data_dir) if f[-4:] == ".txt"]  # grab all text files in datadir
    else:
        datafiles = [args.data_dir]

    dataset = build_dataset(datafiles, tokenizer, args.block_size, num_proc=16)
    dataset = dataset.train_test_split(test_size=0.05)
    logger.info(f"Finished loading dataset with {len(dataset['train'])} training examples and {len(dataset['test'])} testing examples ")

    model = BigramLanguageModel(bpe_vocab_size=args.bpe_vocab_size, num_embeddings=513, block_size=args.block_size, num_heads=args.head_num, num_layers=args.head_num, dropout=0.2)

    training_args = TrainingArguments(  # we can overwrite the trainer, and make use our own optim
        output_dir=args.model_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=1_000,
        save_total_limit=3,
        logging_strategy="epoch",
        disable_tqdm=True,
        evaluation_strategy="steps",
        eval_steps=1_000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    trainer.train() 
    trainer.save_model(args.model_dir)  # will this overwrite? Is it needed?

    # if is_distributed and use_cuda:
    #     # multi-machine multi-gpu case
    #     model = torch.nn.parallel.DistributedDataParallel(model)
    # else:
    #     # single-machine multi-gpu case or single-machine or multi-machine cpu case
    #     model = torch.nn.DataParallel(model)

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)


# def test(model, test_loader, device):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
#             pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     logger.info(
#         "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
#             test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
#         )
#     )

# This is used for running the model on an endpoint for infrencing

def getEncoderFiles(args):
    merges_file = Path(f"{args.merges_file}/merges.txt")
    vocab_file = Path(f"{args.vocab_file}/vocab.json")
    print(f"Files found in vocab: {os.listdir(args.vocab_file)}")
    print(f"Files found in merges: {os.listdir(args.merges_file)}")
    assert merges_file.exists(), f"Merge file not found at: {merges_file.resolve()}"
    assert vocab_file.exists(), f"Vocab file not found at: {vocab_file.resolve()}"
    
    return(merges_file, vocab_file)

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(BigramLanguageModel())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--bpe_vocab_size", type=int, default=7000, metavar="bpe", help="vocab size of the bpe encoder"
    )
    parser.add_argument(
        "--block_size", type=int, default=128, help="Max content length for predictions"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 6500)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--head_num", type=int, default=3, metavar="HN", help="number of heads in the model"
    )
    parser.add_argument(
        "--layer_num", type=int, default=3, metavar="LN", help="number of layers in the model"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS", "[]")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST", 'rip'))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "input_data_files/kant.txt"))
    parser.add_argument("--vocab-file", type=str, default=os.environ.get("SM_CHANNEL_VOCAB", "./tests/KantaiBERT"))
    parser.add_argument("--merges-file", type=str, default=os.environ.get("SM_CHANNEL_MERGES", "./tests/KantaiBERT"))
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS", torch.cuda.device_count()))

    train(parser.parse_args())