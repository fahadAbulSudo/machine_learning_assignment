from src import generate_models
from connection import main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Automation of Employee recommendation"
    )
    #Add the parameters 
    parser.add_argument('hostname', help="EC2 hostID", default="13.232.150.104")
    parser.add_argument('username', help="EC2 Instance Username", default="ubuntu")

    #Parse the arguments
    args = parser.parse_args()
    print(args)

    model = generate_models.generate_models()
    transfer = main.transfer_models(args.hostname, args.username)