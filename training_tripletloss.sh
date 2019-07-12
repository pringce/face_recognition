export PYTHONPATH=PYTHONPATH:$(pwd)/facenet/src

python3 /root/facenet/src/train_tripletloss.py \
--logs_base_dir /root/facenet_trained_model_epoch/logs \
--models_base_dir /root/facenet_trained_model_epoch/models \
--data_dir /root/vggface2_160/ \
--image_size 160 \
--model_def models.inception_resnet_v2_new \
--pretrained_model /root/facenet_trained_model_epoch/models/20190604-104402 \
--lfw_dir /root/lfw/raw_160/ \
--optimizer MOM \
--learning_rate 0.001 \
--max_nrof_epochs 1000 \
--keep_probability 0.8 \
--learning_rate_schedule_file /root/facenet/data/learning_rate_again.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_pairs /root/lfw/pairs.txt
