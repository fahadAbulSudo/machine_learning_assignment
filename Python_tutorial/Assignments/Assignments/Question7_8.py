import mysql.connector

class DbManager:

    def __init__(self, host, user, password, auth_plugin, database):
        self.host = host
        self.user = user
        self.password = password
        self.auth_plugin = auth_plugin
        self.database = database
        self.db_connection = None
        print("Initializing the context manager class")

    def __enter__(self):
        print("We are in enter")
        self.db_connection = mysql.connector.connect(host=self.host, user=self.user, password=self.password,
                                                     auth_plugin=self.auth_plugin, database=self.database)
        return self.db_connection

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.db_connection.close()
        print("Work has been done")


with DbManager('localhost', 'root', 'root', 'mysql_native_password', 'assignment7_8') as db:
    mycursor = db.cursor()
    mycursor.execute("Select name From employee Where salary>=30000;")
    for name in mycursor:
        print(name)


