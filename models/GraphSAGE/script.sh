# graphsage
python gnn.py --log_steps 1 --num_layers 5 --hidden_channels 512 --dropout 0.7 --lr 0.01 --epochs 3000 --runs 5 --device 0 --use_sage

# graphsage + ClusterALL
## k=5
python main_gnn.py --log_steps 1 --num_layers 5 --hidden_channels 512 --dropout 0.7 --lr 0.01 --epochs 3000 --runs 5 --device 0 --num_parts 5 --epoch_gap 199 --use_sage  --dropout_cluster 0.3