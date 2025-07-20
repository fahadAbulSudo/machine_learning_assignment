from io import StringIO
import paramiko
import os
def transfer_models(hostname, username):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #private_key = StringIO("streamlit_buddy1.pem")
    cert = paramiko.RSAKey.from_private_key_file(os.path.abspath("connection/streamlit_buddy1.pem"))
    #pk = paramiko.RSAKey.from_private_key(private_key)
    ssh.connect(hostname=hostname,username=username,pkey=cert,port=22)
    sftp_client = ssh.open_sftp()
    sftp_client.put(os.path.abspath("output/employee_skill.pkl"),"/home/ubuntu/recommend_app/output/employee_skill.pkl")
    sftp_client.put(os.path.abspath("output/similarity.pkl"),"/home/ubuntu/recommend_app/output/similarity.pkl")
    sftp_client.close()
    ssh.close()