CUDA_VISIBLE_DEVICES=0 python train_and_eval.py --train_source_file ../dataset/train/inputs.jsonl.gz --train_target_file ../dataset/train/summaries.jsonl.gz --valid_source_file ../dataset/test/inputs.jsonl.gz --valid_target_file ../dataset/test/summaries.jsonl.gz --node_vocab_file ../dataset/node_5.vocab --edge_vocab_file ../dataset/edge.vocab --target_vocab_file ../dataset/output_5.vocab --train_steps 1000000 --lr_decay_rate 0.95 --lr_decay_steps 10000 --copy_attention --model_name cnn_filter_seed_2020_lr_0.0001 --checkpoint_dir cnn_filter_seed_2020_lr_0.0001 --rnn_hidden_size 256 --rnn_hidden_dropout 0.0 --node_features_dropout 0.0 --validation_interval 5000 --embeddings_dropout 0.2 --learning_rate 0.0001 --batch_size 16 --checkpoint_interval 5000 --beam_width 10 --patience 100 --seed 2020 --ggnn_num_layers 4