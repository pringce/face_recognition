export PYTHONPATH=PYTHONPATH:$(pwd)/facenet/src

python3 /root/facenet/src/train_arcface.py \
--logs_base_dir /root/facenet_trained_model_epoch/logs \
--models_base_dir /root/facenet_trained_model_epoch/models \
--data_dir /root/vggface2_plus_asia_160/ \
--image_size 160 \
--model_def models.inception_resnet_v2_new \
--lfw_dir /root/lfw/raw_160/ \
--center_loss_factor 1 \
--pretrained_model /root/facenet_trained_model_epoch/models/20190620-033445 \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 1000 \
--keep_probability 0.8 \
--learning_rate_schedule_file /root/facenet/data/learning_rate_again.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 0 \
--lfw_pairs /root/lfw/pairs.txt \
--validation_set_split_ratio 0.0 \
--validate_every_n_epochs 10 \
--prelogits_norm_loss_factor 5e-4
