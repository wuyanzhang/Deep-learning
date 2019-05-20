python ./retrain.py ^
--image_dir ./dataset/train ^
--bottleneck_dir bottleneck ^
--how_many_training_steps 200 ^
--model_dir ./inception_model/ ^
--output_graph ./output_graph.pb ^
--output_labels ./output_labels.txt
pause