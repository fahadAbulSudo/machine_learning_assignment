#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# #echo "$SCRIPT_DIR"
# cd $SCRIPT_DIR
# pwd

# get input options download/vectorize/upload
while getopts d:v:u: flag
do
    case "${flag}" in
        d) download=${OPTARG};;
        v) vectorize=${OPTARG};;
        u) upload=${OPTARG};;
    esac
done

SKIP="skip"
A4="a4"
EXT="ext"

# to skip download of resumes
if ! [ -z "$download" ] && [ $download = $SKIP ]
then
echo "Proceed without download.."
else
echo "Downloading files from S3 to local.."
aws s3 sync s3://cv-filtering/input $SCRIPT_DIR/../../profiles/input --delete # local files will be deleted
aws s3 sync s3://cv-filtering/output $SCRIPT_DIR/../../profiles/output --delete # local files will be deleted
fi

# to select vectorization options
if ! [ -z "$vectorize" ] && [ $vectorize = $A4 ]
then
echo "Vectorizing only A4 profiles.."
python3 $SCRIPT_DIR//vectorizeModelA4.py
elif ! [ -z "$vectorize" ] && [ $vectorize = $EXT ]
then
echo "Vectorizing only External profiles.."
python3 $SCRIPT_DIR//vectorizeModelExt.py
elif ! [ -z "$vectorize" ] && [ $vectorize = $SKIP ]
then
echo "Proceed without vectorizing.."
else
echo "Vectorizing all candidate profiles.."
python3 $SCRIPT_DIR//vectorizeModelA4.py
python3 $SCRIPT_DIR//vectorizeModelExt.py
fi

# to skip upload
if ! [ -z "$upload" ] && [ $upload = $SKIP ]
then
echo "Proceed without upload.."
else
echo "Uploading files from local to S3.."
aws s3 sync $SCRIPT_DIR/../../profiles/archive s3://cv-filtering/archive # first sync and add to archive files
aws s3 sync $SCRIPT_DIR/../../profiles/output s3://cv-filtering/output --delete # delete s3 files and sync latest output
fi
