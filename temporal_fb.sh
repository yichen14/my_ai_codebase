python3 entries/train.py --cfg configs/link_pred_cont_temporal_fb.yaml --lr 0.001 --device 0 --ptb_rate 0.1 &
python3 entries/train.py --cfg configs/link_pred_cont_temporal_fb.yaml --lr 0.001 --device 0 --ptb_rate 0.3 &
python3 entries/train.py --cfg configs/link_pred_cont_temporal_fb.yaml --lr 0.001 --device 0 --ptb_rate 0.5 &
python3 entries/train.py --cfg configs/link_pred_cont_temporal_fb.yaml --lr 0.001 --device 0 --ptb_rate 0.7 &

python3 entries/train.py --cfg configs/link_pred_temporal_fb.yaml --device 1 --ptb_rate 0.1 &
python3 entries/train.py --cfg configs/link_pred_temporal_fb.yaml --device 1 --ptb_rate 0.3 &
python3 entries/train.py --cfg configs/link_pred_temporal_fb.yaml --device 1 --ptb_rate 0.5 &
python3 entries/train.py --cfg configs/link_pred_temporal_fb.yaml --device 1 --ptb_rate 0.7 



