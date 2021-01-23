import argparse

# gittest 1
class Hparams:
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=100, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--load_ckpt', default=False, type=bool)

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--seq_len', default=10, type=int,
                        help="length of clip sequence")
    parser.add_argument('--seq_step', default=2, type=int,
                        help='step between ')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")
    parser.add_argument('--gpu_num', default=2, type=int,
                        help='number of gpus')

    # file & path
    parser.add_argument('--label_path', default=r'/data0/hulk/bilibili/label_record_zmn_24s.json',
                        type=str, help='path of label file')
    parser.add_argument('--feature_path',default=r'/data0/hulk/bilibili/feature/',
                        type=str, help='directory of feature file')
    parser.add_argument('--model_save_dir',default=r'/data0/hulk/VHL_GNN/models/',
                        type=str, help='directory of model saving')
    parser.add_argument('--log_dir', default=r'/data0/hulk/VHL_GNN/log/')

# parameters for self-attention model
class Hparams_selfattn:
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--load_ckpt', default=False, type=bool)

    # checkpoint
    parser.add_argument('--ckpt_epoch', default=0, type=int,
                        help="Start to save ckpt")
    parser.add_argument('--ckpt_num', default=30, type=int,
                        help="number of ckpt to keep")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=1, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--seq_len', default=10, type=int,
                        help="length of clip sequence")
    parser.add_argument('--seq_step', default=1, type=int,
                        help='step between ')
    parser.add_argument('--dropout_rate', default=0.9, type=float)
    parser.add_argument('--gpu_num', default=2, type=int,
                        help='number of gpus')

    # file & path
    parser.add_argument('--label_path', default=r'/public/data0/users/hulinkang/bilibili/label_record_zmn_24s.json',
                        type=str, help='path of label file')
    parser.add_argument('--feature_path',default=r'/public/data0/users/hulinkang/bilibili/feature/',
                        type=str, help='directory of feature file')
    parser.add_argument('--model_save_dir',default=r'/public/data0/users/hulinkang/VHL_GNN/models/',
                        type=str, help='directory of model saving')
    parser.add_argument('--log_dir', default=r'/public/data0/users/hulinkang/VHL_GNN/log/')