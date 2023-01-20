import boto3
import json
import pandas as pd
import s3fs
import time
import utils

from datetime import datetime

def create_manifest_from_bucket(bucket, prefix, folder, labels, output_bucket):

    creation_date = str(pd.to_datetime(datetime.now()))[:23].replace(' ','T')
    
    auto_label = {}
    for index, label in enumerate(labels):
        auto_label.update({label: index + 1})

    fs = s3fs.S3FileSystem()
    
    with fs.open(output_bucket + f'/{folder}.manifest', 'w') as f:
        for label in labels:
            files = fs.ls(path=f'{bucket}/{prefix}/{folder}/{label}/', detail=True)
            for file in files:
                if file['Size'] > 0:
                    key = file['Key']
                    manifest_row = {
                        'source-ref': f's3://{key}',
                        'auto-label': auto_label[label],
                        'auto-label-metadata': {
                            'confidence': 1,
                            'job-name': 'labeling-job/auto-label',
                            'class-name': label,
                            'human-annotated': 'yes',
                            'creation-date': creation_date,
                            'type': 'groundtruth/image-classification'
                        }
                    }
                    f.write(json.dumps(manifest_row, indent=None) + '\n')
                    
def start_model(project_arn, model_arn, version_name, min_inference_units=1):
    client = boto3.client('rekognition')
    try:
        print('Startowanie modelu: ' + model_arn)
        response = client.start_project_version(ProjectVersionArn=model_arn, MinInferenceUnits=min_inference_units)
        
        project_version_running_waiter = client.get_waiter('project_version_running')
        project_version_running_waiter.wait(ProjectArn=project_arn, VersionNames=[version_name])

        describe_response=client.describe_project_versions(ProjectArn=project_arn, VersionNames=[version_name])
        for model in describe_response['ProjectVersionDescriptions']:
            print("Status: " + model['Status'])
            print("Info: " + model['StatusMessage'])
            
    except Exception as e:
        print(e)
        
    
def stop_model(model_arn):

    print('Stopping model:' + model_arn)

    try:
        reko = boto3.client('rekognition')
        response = reko.stop_project_version(ProjectVersionArn=model_arn)
        status = response['Status']
        print('Status: ' + status)
        
    except Exception as e:  
        print(e)  

    
def show_custom_labels(model, bucket, image, min_confidence):
    reko = boto3.client('rekognition')
    try:
        response = reko.detect_custom_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': image}},
            MinConfidence=min_confidence,
            ProjectVersionArn=model
        )
        
    except Exception as e:
        print(f'Exception encountered when processing {image}')
        print(e)
        
    return response['CustomLabels']

def get_results(project_version_arn, bucket, s3_path, label=None, verbose=True):
    fs = s3fs.S3FileSystem()
    data = {}
    counter = 0
    predictions = pd.DataFrame(columns=['image', 'normal', 'abnormal'])
    
    for file in fs.ls(path=s3_path, detail=True, refresh=True):
        if file['Size'] > 0:
            image = '/'.join(file['Key'].split('/')[1:])
            if verbose == True: print('.', end='')

            labels = show_custom_labels(project_version_arn, bucket, image, 0.0)
            for L in labels:
                data[L['Name']] = L['Confidence']
                
            predictions = predictions.append(pd.Series({
                'image': file['Key'].split('/')[-1],
                'abnormal': data['abnormal'],
                'normal': data['normal'],
                'ground truth': label
            }), ignore_index=True)
            
            counter += 1
            if counter % 100 == 0:
                if verbose == True: print('|', end='')
                time.sleep(1)
            
    return predictions

def reshape_results(df, unknown_threshold=50.0):

    new_val_predictions = pd.DataFrame(columns=['Image', 'Ground Truth', 'Prediction', 'Confidence Level'])

    for index, row in df.iterrows():
        new_row = pd.Series(dtype='object')
        new_row['Image'] = row['image']
        new_row['Ground Truth'] = row['ground truth']
        if row['normal'] >= unknown_threshold:
            new_row['Prediction'] = 'normal'
            new_row['Confidence Level'] = row['normal'] / 100

        elif row['abnormal'] >= unknown_threshold:
            new_row['Prediction'] = 'abnormal'
            new_row['Confidence Level'] = row['abnormal'] / 100

        else:
            new_row['Prediction'] = 'unknown'
            new_row['Confidence Level'] = 0.0

        new_val_predictions = new_val_predictions.append(pd.Series(new_row), ignore_index=True)

    return new_val_predictions
