# Image-Processing-on-Dermoscopic-Skin-Lesions

# Image-Processing-on-Dermoscopic-Skin-Lesions

## run the following if images folder is empty:
pip install requests
python download_isic.py



python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python run_preprocessing.py --limit 20        # small test run
python visualize_preprocessing.py             # check random samples
python run_preprocessing.py --overwrite --limit 20   # rerun after tuning config.py
python run_preprocessing.py                   # full run (resumes if interrupted)
python visualize_preprocessing.py --mode hairiest    # inspect the hairiest images

# after segmentation
pip install -r requirements.txt
python run_segmentation.py --limit 20                      # small test; prints GT sources, label counts, timings
python visualize_segmentation.py --n 5 --label malignant
python visualize_segmentation.py --n 5 --label benign
python visualize_segmentation.py --mode worst --n 6        # the failures
python run_segmentation.py                                 
