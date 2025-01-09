# DATASETS="nytimes-256-angular gist-960-euclidean glove-100-angular glove-25-angular mnist-784-euclidean sift-128-euclidean"
DATASETS="
nytimes-256-angular
glove-100-angular
glove-25-angular
mnist-784-euclidean
sift-128-euclidean
dbpedia-openai-100k-angular
"
# DATASETS="mnist-784-euclidean"
# DATASETS="gist-960-euclidean"
# DATASETS="glove-25-angular"
# DATASETS="sift-128-euclidean"
# DATASETS="random-xs-20-angular"
# DATASETS="random-xs-20-euclidean"
# DATASETS="random-s-100-euclidean"
# DATASETS="random-xs-16-hamming"
# DATASETS="random-s-128-hamming"
# DATASETS="sift-256-hamming"
# DATASETS="word2bits-800-hamming"
# DATASETS="dbpedia-openai-100k-angular"
# DATASETS="kosarak-jaccard"

# for DATASET in $DATASETS
# do
#     wget https://ann-benchmarks.com/$DATASET.hdf5 -O data/$DATASET.hdf5
# done

for DATASET in $DATASETS
do
    echo "Running $DATASET"
    python3 run.py --dataset=$DATASET --algorithm=semadb --runs=1 --local --force
    # python3 plot.py --dataset=$DATASET
done

for DATASET in $DATASETS
do
    echo "Plotting $DATASET"
    python3 plot.py --dataset=$DATASET
done

# python3 data_export.py --out res.csv
