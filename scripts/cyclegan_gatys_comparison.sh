mkdir temp/comparison -p

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e-5 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys0.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e-4 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys1.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e-3 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys2.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e-2 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys3.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e-1 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys4.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e0 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys5.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e1 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys6.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e2 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys7.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e3 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys8.png

python scripts/train_gatys.py --content-img "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --style-img data/monet2photo/trainA/00001.jpg --content-weight 1e4 --style-weight 1e4 --epochs 200 --display-image --save-image temp/comparison/gatys9.png

python scripts/inference_cyclegan.py --model "models\regularization\lightning_logs\cyclegan_regularization_experiment_l0\version_3\checkpoints\checkpoint_periodic-epoch=0029.ckpt" --image "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --output-image temp/comparison/cycle0.png

python scripts/inference_cyclegan.py --model "models\regularization\lightning_logs\cyclegan_regularization_experiment_l0.5\version_3\checkpoints\checkpoint_periodic-epoch=0029.ckpt" --image "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --output-image temp/comparison/cycle1.png

python scripts/inference_cyclegan.py --model "models\regularization\lightning_logs\cyclegan_regularization_experiment_l1\version_3\checkpoints\checkpoint_periodic-epoch=0029.ckpt" --image "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --output-image temp/comparison/cycle2.png

python scripts/inference_cyclegan.py --model "models\regularization\lightning_logs\cyclegan_regularization_experiment_l2\version_3\checkpoints\checkpoint_periodic-epoch=0029.ckpt" --image "data/monet2photo/trainB/2013-11-08 16_45_24.jpg" --output-image temp/comparison/cycle3.png
