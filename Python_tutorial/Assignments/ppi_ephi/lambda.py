import boto3

def get_instance_name(ec2_client, instance_id):
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        tags = instance.get('Tags', [])
        for tag in tags:
            if tag['Key'] == 'Name':
                return tag['Value']
        return instance_id
    except Exception as e:
        return "Error retrieving instance name: {}".format(str(e))

def lambda_handler(event, context):
    sns_topic_arn = 'arn:aws:sns:ap-south-1:567676415231:EC2notice' 
    sns_client = boto3.client('sns')
    
    event_source = event['source']
    event_detail = event['detail']
    
    aws_region = event_detail['awsRegion']

    if event_source == "aws.s3":
        user = event_detail['userIdentity']['arn']
        source = event_source
        s3_event = event["detail"]["eventName"]
        s3_bucket =event["detail"]["requestParameters"]["bucketName"]
        message_data = {
                    "user":user,
                    "resource":source,
                    "event_type":s3_event,
                    "event_resource":s3_bucket	
                }

        message = f"Bucket Details: {message_data}"
        sns_subject = 'Bucket Details Event Notification'

    elif event_source == 'aws.ec2':
        ec2_client = boto3.client('ec2', region_name=aws_region) 
        instance_id = event_detail['requestParameters']['instancesSet']['items'][0]['instanceId']
        instance_name = get_instance_name(ec2_client, instance_id)
        user = event_detail['userIdentity']['arn']
        source = event_source
        ec2_event = event_detail['eventName']
        message_data = {
                    "user":user,
                    "resource":source,
                    "event_type":ec2_event,
                    "event_resource":instance_name	
                }
        message = f"EC2 instance Details: {message_data}"
        sns_subject = 'EC2 instance Details Event Notification'
        
    elif event_source == 'aws.lambda':
        instance_name = event_detail['requestParameters']['functionName']
        user = event_detail['userIdentity']['arn']
        source = event_source
        lambda_event = event_detail['eventName']
        message_data = {
                    "user":user,
                    "resource":source,
                    "event_type":lambda_event,
                    "event_resource":instance_name	
                }
        message = f"Lambda function Details: {message_data}"
        sns_subject = 'Lambda function Details Event Notification'

    else:
        user = event_detail['userIdentity']['arn']
        source = event_source
        event = event_detail['eventName']
        message_data = {
                    "user":user,
                    "resource":source,
                    "event_type":event	
                }
        message = f"Event Details: {message_data}"
        sns_subject = 'Event Notification'
   
    
    #sns_subject = 'EC2/Lambda Event Notification'
    
    sns_client.publish(
        TopicArn=sns_topic_arn,
        Subject=sns_subject,
        Message=message
    )

    return {
        'statusCode': 200,
        'body': 'Notification sent successfully'
    }


##**************************************OLD CODE******************************************************8

import boto3

def get_instance_name(ec2_client, instance_id):
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        tags = instance.get('Tags', [])
        for tag in tags:
            if tag['Key'] == 'Name':
                return tag['Value']
        return instance_id
    except Exception as e:
        return "Error retrieving instance name: {}".format(str(e))

def lambda_handler(event, context):
    sns_topic_arn = 'arn:aws:sns:ap-south-1:567676415231:EC2notice' 
    sns_client = boto3.client('sns')
    
    event_source = event['source']
    event_detail = event['detail']
    
    aws_region = event_detail['awsRegion']

    if event_source == "aws.s3":
        user = event_detail['userIdentity']['arn']
        source = 'S3'
        s3_event = event["detail"]["eventName"]
        s3_bucket =event["detail"]["requestParameters"]["bucketName"]
        if s3_event.startswith('CreateBucket'):
            message_data = {
                    	"user":user,
                    	"resource":source,
                    	"event_type":s3_event,
                    	"event_resource":s3_bucket	
                    }

            message = f"Created Bucket Details: {message_data}"
            sns_subject = 'Bucket Creation Event Notification'
        elif s3_event.startswith('DeleteBucket'):
            message_data = {
                    	"user":user,
                    	"resource":source,
                    	"event_type":s3_event,
                    	"event_resource":s3_bucket	
                    }

            message = f"Deleted Bucket Details: {message_data}"
            sns_subject = 'Bucket Deletion Event Notification'
    elif event_source == 'aws.ec2':
        ec2_client = boto3.client('ec2', region_name=aws_region) 
        instance_id = event_detail['requestParameters']['instancesSet']['items'][0]['instanceId']
        instance_name = get_instance_name(ec2_client, instance_id)
        if event_detail['eventName'] == 'StartInstances':
            user = event_detail['userIdentity']['arn']
            source = 'ec2'
            ec2_event = event_detail['eventName']
            message_data = {
                    	"user":user,
                    	"resource":source,
                    	"event_type":ec2_event,
                    	"event_resource":instance_name	
                    }
            message = f"Started EC2 instance Details: {message_data}"
            sns_subject = 'EC2 instance start Event Notification'
        elif event_detail['eventName'] == 'StopInstances':
            user = event_detail['userIdentity']['arn']
            source = 'ec2'
            ec2_event = event_detail['eventName']
            message_data = {
                    	"user":user,
                    	"resource":source,
                    	"event_type":ec2_event,
                    	"event_resource":instance_name	
                    }
            message = f"Stoped EC2 instance Details: {message_data}"
            sns_subject = 'EC2 instance stop Event Notification'
        elif event_detail['eventName'] == 'TerminateInstances':
            user = event_detail['userIdentity']['arn']
            source = 'ec2'
            ec2_event = event_detail['eventName']
            message_data = {
                    	"user":user,
                    	"resource":source,
                    	"event_type":ec2_event,
                    	"event_resource":instance_name	
                    }
            message = f"Terminated EC2 instance Details: {message_data}"
            sns_subject = 'EC2 instance terminate Event Notification'
        elif event_detail['eventName'] == 'RunInstances':
            user = event_detail['userIdentity']['arn']
            source = 'ec2'
            ec2_event = event_detail['eventName']
            message_data = {
                    	"user":user,
                    	"resource":source,
                    	"event_type":ec2_event,
                    	"event_resource":instance_name	
                    }
            message = f"Launched EC2 instance Details: {message_data}"
            sns_subject = 'EC2 instance launch Event Notification'
    elif event_source == 'aws.lambda':
        instance_name = event_detail['requestParameters']['functionName']
        if event_detail['eventName'] == 'CreateFunction20150331':
            user = event_detail['userIdentity']['arn']
            source = 'Lambda'
            ec2_event = event_detail['eventName']
            message_data = {
                    	"user":user,
                    	"resource":source,
                    	"event_type":ec2_event,
                    	"event_resource":instance_name	
                    }
            message = f"Lambda function Creation Details: {message_data}"
            sns_subject = 'Lambda function created Event Notification'
        elif event_detail['eventName'] == 'DeleteFunction20150331':
            user = event_detail['userIdentity']['arn']
            source = 'Lambda'
            ec2_event = event_detail['eventName']
            message_data = {
                    	"user":user,
                    	"resource":source,
                    	"event_type":ec2_event,
                    	"event_resource":instance_name	
                    }
            message = f"Lambda function Deletion Details: {message_data}"
            sns_subject = 'Lambda function deleted Event Notification'
    else:
        instance_name = "Unknown"
        message = "Unknown event for {}.".format(instance_name)
   
    
    #sns_subject = 'EC2/Lambda Event Notification'
    
    sns_client.publish(
        TopicArn=sns_topic_arn,
        Subject=sns_subject,
        Message=message
    )

    return {
        'statusCode': 200,
        'body': 'Notification sent successfully'
    }