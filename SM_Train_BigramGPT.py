import argparse
import json
import logging
import os
import sys
from functools import partial

#import sagemaker_containers
import torch
import torch.distributed as dist
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
# from torchvision import datasets, transforms

from torch.utils.data import IterableDataset
from torch.utils.data import Dataset


#from gpt_tokenizers import BytePairEncoder
from transformers import LineByLineTextDataset
from transformers import GPT2TokenizerFast
from transformers import GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling

from transformers.tokenization_utils import PreTrainedTokenizer

from BigramGPT import BigramLanguageModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def _get_train_data_loader(batch_size, file, block_size, encoder, is_distributed, **kwargs):
    logger.info("Get train data loader")
    dataset = orcaDataset(
        file,
        block_size,
        encoder
    )

    # dataset = LineByLineTextDataset(
    #     #tokenizer=encoder,
    #     file_path=file,
    #     block_size=block_size
    # )

    collate_fn = DataCollatorForLanguageModeling(
        tokenizer=encoder, mlm=False
    )
    # train_sampler = (
    #     torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    # )
    print(f"Batj size: {batch_size} returning dataloader 3")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        # shuffle=train_sampler is None,
        # sampler=train_sampler,
        **kwargs
    )

# class orcaDataset(IterableDataset):
#     def __init__(self, file, block_size, tokenizer: PreTrainedTokenizer):
#         self.tokenizer = tokenizer
#         self.file = file
#         self.block_size = block_size

#     # Do we want to use and count overlapping sentences?
#     def __iter__(self):
#         with open(self.file, encoding="utf-8") as f:
#             lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

#         encoding = partial(self.tokenizer, add_special_tokens=True, truncation=True, max_length=self.block_size)
#         mapped_iter = map(encoding, iter(lines))
#         return mapped_iter

class orcaDataset(Dataset):
    def __init__(self, file, block_size, tokenizer):
        with open(file, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]


        encoding = partial(tokenizer, add_special_tokens=True, truncation=True, max_length=block_size)
        self.mapped_iter = map(encoding, iter(self.lines))
        self.last = None

        

    def __len__(self):
        return len(self.lines)  # this wont ever update

    def __getitem__(self, idx):
        if not self.last:
            self.last = next(self.mapped_iter)

        nextval = next(self.mapped_iter)
        val = (self.last, nextval)
        self.last = next

        return (val)

    # Do we want to use and count overlapping sentences?
    # def __getitem__(self, idx):  
    #     #ix = torch.randint(len(self.data) - self.block_size, (self.block_size,)) this is just a random select right?
    #     ix = idx * self.block_size
    #     x = torch.stack([self.data[i:i + self.block_size] for i in ix])  # will this be called out or range and crash?
    #     y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])

# def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
#     logger.info("Get test data loader")
#     return torch.utils.data.DataLoader(
#         datasets.MNIST(
#             training_dir,
#             train=False,
#             transform=transforms.Compose(
#                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#             ),
#         ),
#         batch_size=test_batch_size,
#         shuffle=True,
#         **kwargs
#     )


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # logger.info("training encoder")
    # byte_pair_encoder = BytePairEncoder(args.bpe_vocab_size, 2)
    # byte_pair_encoder.train(args.data_dir)

    # logger.info("saving encoder")
    # byte_pair_encoder.save(args.model_dir, "Orca_encoder")

    # tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')  # will this work??
    print("1 defined toke")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token="<pad>")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, args.block_size, tokenizer, is_distributed, **kwargs)
    #test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    # logger.debug(
    #     "Processes {}/{} ({:.0f}%) of train data".format(
    #         len(train_loader.sampler),
    #         len(train_loader.dataset),
    #         100.0 * len(train_loader.sampler) / len(train_loader.dataset),
    #     )
    # )

    # logger.debug(
    #     "Processes {}/{} ({:.0f}%) of test data".format(
    #         len(test_loader.sampler),
    #         len(test_loader.dataset),
    #         100.0 * len(test_loader.sampler) / len(test_loader.dataset),
    #     )
    # )

    logger.info("Finished loading dataset")

    model = BigramLanguageModel().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)  # RMEMEBER HIS CODE LOADS IN THE LOADER
            optimizer.zero_grad()
            output, loss = model(data, target)
            #loss = F.nll_loss(output, target)
            #loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        #loss.item(),
                        loss,
                    )
                )
        # test(model, test_loader, device)
    save_model(model, args.model_dir)

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
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--bpe_vocab_size", type=int, default=6500, metavar="bpe", help="vocab size of the bpe encoder"
    )
    parser.add_argument(
        "--block_size", type=int, default=256, help="Max content length for predictions"
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
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "input_data_files/cleaned_orca_dataset.txt"))
    parser.add_argument("--num-gpus", type=int, default=os.environ.get("SM_NUM_GPUS", torch.cuda.device_count()))

    train(parser.parse_args())