# Tetris Hu Moments Classifier

Three scripts:
- generator.py
- trainer.py
- classifier.py

## Setup
python -m pip install -r requirements.txt

## Labels
labels.json uses 1-based labels:
1=I, 2=O, 3=T, 4=S, 5=Z, 6=J, 7=L

## Generate dataset
python generator.py --label 1 --output dataset.csv
Press SPACE to capture Hu moments.

## Train
python trainer.py --data dataset.csv --model model.joblib --test-split 0.2

## Classify
python classifier.py --model model.joblib --labels labels.json

## Notes
- Use --invert if the background is darker than the shapes.
- Use --raw-hu if you do not want the log transform.
- Use `--edges` to use Canny edges + dilation which can improve
	contour detection when simple thresholding leaves gaps in the shape border.
