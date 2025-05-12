import argparse
import model as M
import nnue_dataset
import pytorch_lightning as pl
import features
import os
import sys
import torch
torch.set_float32_matmul_precision('medium')
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

def make_data_loaders(train_filenames, val_filenames, feature_set, num_workers, batch_size, filtered, random_fen_skipping, wld_filtered, early_fen_skipping, param_index, main_device, epoch_size, val_size):
  # Epoch and validation sizes are arbitrary
  features_name = feature_set.name
  train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filenames, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, wld_filtered=wld_filtered, early_fen_skipping=early_fen_skipping, param_index=param_index, device=main_device)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filenames, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, wld_filtered=wld_filtered, early_fen_skipping=early_fen_skipping, param_index=param_index, device=main_device)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def flatten_once(lst):
    return sum(lst, [])

def main():
  parser = argparse.ArgumentParser(description="Trains the network.")
  parser.add_argument("datasets", action='append', nargs='+', help="Training datasets (.binpack). Interleaved at chunk level if multiple specified. Same data is used for training and validation if not validation data is specified.")

  # === THÊM CÁC THAM SỐ CỦA TRAINER VÀO PARSER MỘT CÁCH THỦ CÔNG ===
  # Tham khảo tài liệu của phiên bản PyTorch Lightning bạn đang dùng để biết tên và giá trị mặc định chính xác.
  parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator to use ('cpu', 'gpu', 'tpu', 'mps', 'auto')")
  parser.add_argument("--devices", type=str, default="auto", help="Devices to use (e.g., 1, '0', '0,1', 'auto')")
  parser.add_argument("--strategy", type=str, default=None, help="Distributed strategy (e.g., 'ddp', 'fsdp')") # Hoặc 'auto'
  parser.add_argument("--num_nodes", type=int, default=1, help="Number of C_NODES for distributed training")
  parser.add_argument("--max_epochs", type=int, default=None, help="Stop training once this number of epochs is reached.") # Thay vì or 800 ở dưới
  parser.add_argument("--min_epochs", type=int, default=None, help="Force training for at least these many epochs.")
  parser.add_argument("--max_steps", type=int, default=-1, help="Stop training after this number of steps. Disabled by default (-1).")
  parser.add_argument("--min_steps", type=int, default=None, help="Force training for at least these many steps.")
  parser.add_argument("--precision", type=str, default="32-true", help="Double precision (64), full precision (32), 16bit mixed precision (16-mixed), Brain Floating Point (bf16-mixed).")
  parser.add_argument("--default_root_dir", type=str, default=None, help="Default path for logs and weights. If not set, a directory ``lightning_logs`` will be created.")
  parser.add_argument("--deterministic", type=str2bool, nargs='?', const=True, default=False, help="Set to True for deterministic training.")
  parser.add_argument("--benchmark", type=str2bool, nargs='?', const=True, default=False, help="Set to True to enable cudnn.benchmark.") # Mặc định có thể là False
  parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulates grads every k batches.")
  parser.add_argument("--gradient_clip_val", type=float, default=None, help="Norm to clip gradients to.")
  parser.add_argument("--val_check_interval", type=float, default=1.0, help="How often to check the validation set.")
  parser.add_argument("--profiler", type=str, default=None, help="To profile individual Lightning hooks.")
  parser.add_argument("--enable_checkpointing", type=str2bool, default=True, help="If ``True``, enable checkpointing.") # Mặc định là True
  # Thêm các tham số khác của Trainer mà bạn muốn điều khiển từ dòng lệnh

  # Các tham số gốc của script
  parser.add_argument("--validation-data", type=str, action='append', nargs='+', dest='validation_datasets', help="Validation data to use for validation instead of the training data.")
  parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
  parser.add_argument("--start-lambda", default=None, type=float, dest='start_lambda', help="lambda to use at first epoch.")
  parser.add_argument("--end-lambda", default=None, type=float, dest='end_lambda', help="lambda to use at last epoch.")
  parser.add_argument("--gamma", default=0.992, type=float, dest='gamma', help="Multiplicative factor applied to the learning rate after every epoch.")
  parser.add_argument("--lr", default=8.75e-4, type=float, dest='lr', help="Initial learning rate.")
  parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Number of worker threads to use for data loading. Currently only works well for binpack.")
  parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
  parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
  parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
  # --smart-fen-skipping bị bỏ qua, chỉ dùng --no-smart-fen-skipping
  parser.add_argument("--no-smart-fen-skipping", action='store_true', dest='no_smart_fen_skipping', help="If used then no smart fen skipping will be done. By default smart fen skipping is done.")
  parser.add_argument("--no-wld-fen-skipping", action='store_true', dest='no_wld_fen_skipping', help="If used then no wld fen skipping will be done. By default wld fen skipping is done.")
  parser.add_argument("--random-fen-skipping", default=3, type=int, dest='random_fen_skipping', help="skip fens randomly on average random_fen_skipping before using one.")
  parser.add_argument("--resume-from-model", dest='resume_from_model', help="Initializes training using the weights from the given .pt model")
  parser.add_argument("--network-save-period", type=int, default=20, dest='network_save_period', help="Number of epochs between network snapshots. None to disable.") # Sẽ dùng cho every_n_epochs của ModelCheckpoint
  parser.add_argument("--save-last-network", type=str2bool, default=True, dest='save_last_network', help="Whether to always save the last produced network.") # Dùng cho save_last của ModelCheckpoint
  parser.add_argument("--epoch-size", type=int, default=100000000, dest='epoch_size', help="Number of positions per epoch.")
  parser.add_argument("--validation-size", type=int, default=1000000, dest='validation_size', help="Number of positions per validation step.")
  parser.add_argument("--param-index", type=int, default=0, dest='param_index', help="Indexing for parameter scans.")
  parser.add_argument("--early-fen-skipping", type=int, default=-1, dest='early_fen_skipping', help="Skip n plies from the start.")
  features.add_argparse_args(parser) # Giữ lại
  args = parser.parse_args()

  args.datasets = flatten_once(args.datasets)
  if args.validation_datasets:
    args.validation_datasets = flatten_once(args.validation_datasets)
  else:
    args.validation_datasets = []

  for dataset in args.datasets:
    if not os.path.exists(dataset):
      raise Exception('{0} does not exist'.format(dataset))

  for val_dataset in args.validation_datasets:
    if not os.path.exists(val_dataset):
      raise Exception('{0} does not exist'.format(val_dataset))

  train_datasets = args.datasets
  val_datasets = train_datasets
  if len(args.validation_datasets) > 0:
    val_datasets = args.validation_datasets

  if (args.start_lambda is not None) != (args.end_lambda is not None):
    raise Exception('Either both or none of start_lambda and end_lambda must be specified.')

  feature_set = features.get_feature_set_from_name(args.features)

  start_lambda = args.start_lambda if args.start_lambda is not None else args.lambda_
  end_lambda = args.end_lambda if args.end_lambda is not None else args.lambda_
  max_epoch = args.max_epochs or 800
  # if args.resume_from_model is None:
  nnue = M.NNUE(
      feature_set=feature_set,
      start_lambda=start_lambda,
      max_epoch=max_epoch,
      end_lambda=end_lambda,
      gamma=args.gamma,
      lr=args.lr,
      param_index=args.param_index
  )
  # else:
  #   nnue = torch.load(args.resume_from_model)
  #   nnue.set_feature_set(feature_set)
  #   nnue.start_lambda = start_lambda
  #   nnue.end_lambda = end_lambda
  #   nnue.max_epoch = max_epoch
  #   # we can set the following here just like that because when resuming
  #   # from .pt the optimizer is only created after the training is started
  #   nnue.gamma = args.gamma
  #   nnue.lr = args.lr
  #   nnue.param_index=args.param_index

  print("Feature set: {}".format(feature_set.name))
  print("Num real features: {}".format(feature_set.num_real_features))
  print("Num virtual features: {}".format(feature_set.num_virtual_features))
  print("Num features: {}".format(feature_set.num_features))

  print("Training with: {}".format(train_datasets))
  print("Validating with: {}".format(val_datasets))

  pl.seed_everything(args.seed)
  print("Seed {}".format(args.seed))

  batch_size = args.batch_size
  if batch_size <= 0:
    batch_size = 16384
  print('Using batch size {}'.format(batch_size))

  print('Smart fen skipping: {}'.format(not args.no_smart_fen_skipping))
  print('WLD fen skipping: {}'.format(not args.no_wld_fen_skipping))
  print('Random fen skipping: {}'.format(args.random_fen_skipping))
  print('Skip early plies: {}'.format(args.early_fen_skipping))
  print('Param index: {}'.format(args.param_index))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)
  tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name="", version="", log_graph=True) # Để PL tự quản lý version
  callbacks_list = []
  if args.enable_checkpointing: # enable_checkpointing giờ là từ parser
      checkpoint_callback = pl.callbacks.ModelCheckpoint(
          save_last=args.save_last_network,
          every_n_epochs=args.network_save_period if args.network_save_period > 0 else None, # Nếu 0 hoặc None, PL có thể có hành vi mặc định
          save_top_k=-1 if args.network_save_period > 0 else 1, # Lưu tất cả nếu lưu theo period, ngược lại lưu top 1
          monitor="val_loss" if args.network_save_period <=0 else None # Cần monitor nếu save_top_k > 0 và không phải -1
      )
      callbacks_list.append(checkpoint_callback)
  # checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=args.save_last_network, every_n_epochs=args.network_save_period, save_top_k=-1)
  trainer = pl.Trainer(
      accelerator=args.accelerator,
      devices=args.devices,
      strategy=args.strategy or "auto",
      num_nodes=args.num_nodes,
      max_epochs=args.max_epochs,
      min_epochs=args.min_epochs,
      max_steps=args.max_steps,
      min_steps=args.min_steps,
      precision=args.precision,
      default_root_dir=args.default_root_dir, # Trainer sẽ sử dụng điều này nếu logger không chỉ định save_dir
      deterministic=args.deterministic,
      benchmark=args.benchmark,
      accumulate_grad_batches=args.accumulate_grad_batches,
      gradient_clip_val=args.gradient_clip_val,
      val_check_interval=args.val_check_interval,
      profiler=args.profiler,
      enable_checkpointing=args.enable_checkpointing, # Điều khiển việc ModelCheckpoint có được dùng không (mặc dù ta tự thêm)
      callbacks=callbacks_list,
      logger=tb_logger
      # Thêm các tham số khác từ args nếu bạn đã định nghĩa chúng trong parser
  )
  # Lọc bỏ các giá trị None để không ghi đè mặc định của Trainer một cách không cần thiết
  main_device = str(trainer.strategy.root_device) if trainer.strategy.root_device else "cpu"
  nnue.to(device=main_device)

  print('Using c++ data loader')
  train, val = make_data_loaders(
    train_datasets,
    val_datasets,
    feature_set,
    args.num_workers,
    batch_size,
    not args.no_smart_fen_skipping,
    args.random_fen_skipping,
    not args.no_wld_fen_skipping,
    args.early_fen_skipping,
    args.param_index,
    main_device,
    args.epoch_size,
    args.validation_size)
  ckpt_path_to_resume = args.resume_from_model if args.resume_from_model else None
  # trainer.fit(nnue, train, val, ckpt_path_to_resume) # Bây giờ trainer là một đối tượng pl.Trainer
  # Ở cuối main()
  ckpt_path_to_resume = args.resume_from_model if args.resume_from_model else None

  # SỬA ĐỔI DÒNG NÀY:
  trainer.fit(
      model=nnue,
      train_dataloaders=train,
      val_dataloaders=val,
      ckpt_path=ckpt_path_to_resume
  )
  with open(os.path.join(logdir, tb_logger.name, f"version_{tb_logger.version}", 'training_finished.txt'), 'w') as f: # Ghi vào thư mục version của logger
    f.write("Training finished.")

if __name__ == '__main__':
  main()
  if sys.platform == "win32":
    os.system(f'wmic process where processid="{os.getpid()}" call terminate >nul')
