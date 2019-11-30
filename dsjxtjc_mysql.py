import pymysql
import pandas as pd
import numpy as np
import time

class mmySQL:
    def __init__(self,hosts_= None,user_ = None,pass_word_=None,
                 database_= None,table_ = None,charset_='utf8'):
        if hosts_==None or user_==None or pass_word_==None or database_==None:
            print('Not initial hosts or user or password or database!')
        self.con = pymysql.connect(host = hosts_,user=user_ ,password=pass_word_,
                               database = database_, charset=charset_)
        self.cursor = self.con.cursor()
        print("database:",database_)

    def show_tables(self):
        self.cursor.execute("SHOW TABLES;")
        tables_name = self.cursor.fetchall()
        tables_name_ = []
        for i in tables_name:
            tables_name_.append(i[0])
        return tables_name_

    def creat_table(self):  # 最好直接在mysql命令行里操作\n",
        create_table_order = input('Please input create table order:')
        if create_table_order ==None or "CREATE TABLE" not in create_table_order:
            print("please pass creat table order")
            return
        self.cursor.execute(create_table_order)

    def insert(self,insert_order = None): # 插入语句\n",
        if insert_order == None or "INSERT INTO" not in insert_order:
            print("please input right insert order")
            return
        try:
            self.cursor.execute(insert_order)
            self.con.commit()
            print(insert_order)
        except:
            self.con.rollback()
            print('insert fail!')

    def change(self,change_order):
        if change_order == None or "UPDATE" not in change_order:
            print("please input right insert order")
            return
        try:
            self.cursor.execute(change_order)
            self.con.commit()
            print(change_order)
        except:
            self.con.rollback()

    def get_all_datas(self,table = None):
        if table == None:
            print('set a table!')
        get_all_data_order =  "SELECT * from %s" % (table)
        self.cursor.execute(get_all_data_order)
        results = self.cursor.fetchall()
        return results

    def select_data(self,order):
        self.cursor.execute(order)
        results = self.cursor.fetchall()
        return results

    def close_sql(self):
        self.con.close()


