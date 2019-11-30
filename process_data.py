import time
import pandas as pd
import numpy as np
from dsjxtjc_mysql import mmySQL
from tqdm import tqdm
import os
from tqdm import tqdm
import random
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
def get_sales_order(sql_obj,table_name):  # 菜品出现次数统计
    all_data = sql_obj.get_all_datas(table_name)
    all_data_list = []
    t1 = time.time()
    for i in all_data:
        all_data_list.append(list(i))
    all_data_list = pd.DataFrame(all_data_list)
    # print('菜品',all_data_list.iloc[:,4].value_counts())
    all_data_list.iloc[:, 8].value_counts().to_csv('./tongji.csv')
    print(time.time() - t1)


def get_sales_order_in_diffclass(sales_order_csv_dir):  # 每类菜品中的排名统计 并统计了每类菜数量比例
    sales_order_csv = pd.read_csv(sales_order_csv_dir, dtype=str,
                                  header=None, index_col=None).values
    dishs_after_classify = []  # 里边存10个数组 一共有10种类型的菜
    for _ in range(10):
        dishs_after_classify.append([])

    l = len(sales_order_csv)
    for i in range(l):
        if len(sales_order_csv[i, 0]) == 5:  # 5位菜号第一位是类号
            class_num = int(sales_order_csv[i, 0][0])
        elif len(sales_order_csv[i, 0]) == 6:  # 6位菜号前两位是类号
            class_num = int(sales_order_csv[i, 0][0:2])
        else:
            print('dishes number is wrong!')
            raise NameError
        dishs_after_classify[class_num-1].append(sales_order_csv[i])

    for i in range(len(dishs_after_classify)):
        save_csv = pd.DataFrame(dishs_after_classify[i])
        save_csv.to_csv(
            './sales_order_in_different_class/class{}.csv'.format(i+1),
            header=None, index=None)
    # 统计每每类菜所占比例
    each_class_num = np.zeros(10)
    for k, each_class in enumerate(dishs_after_classify):
        print(each_class)
        for i in each_class:
            each_class_num[k] += float(i[1])
    print(each_class_num/sum(each_class_num))


def delete_waimai(sql_obj, source_table, target_table):  # 删除外卖订单
    # !!!!!!!!需要提前建一张空表(target_table) 用来存储去除外卖之后的数据
    table_names = sql_obj.show_tables()
    # 保证有target_table
    if target_table not in table_names:
        print('no target_table!')
        raise NameError
    # 保证target_table为空
    target_table_data = sql_obj.select_data("SELECT * from %s" % (target_table))
    if target_table_data:
        print('target_table 非空')
        raise NameError
    # 去除是外卖方式的单子
    select_order = "SELECT * from %s WHERE table_area_name != '外卖'" % (source_table)
    source_datas = sql_obj.select_data(select_order)
    # 程序插入太慢 1000条73s 利用先生成.sql文件再手动导入
    '''for a_data in source_datas:
        print(a_data)
        insert_order = "INSERT INTO {} VALUES {};".format(target_table, a_data)
        sql_obj.insert(insert_order)'''
    with open(target_table+".sql","w",encoding='utf-8') as f:
        for a_data in source_datas:
            insert_order = "INSERT INTO {} VALUES {};\n".format(target_table, a_data)
            f.write(insert_order)


def generate_table_by_order(sql_obj, source_table, target_table): # 按订单号排序
    table_names = sql_obj.show_tables()
    # 保证有target_table
    if target_table not in table_names:
        print('no target_table!')
        raise NameError
    # 保证target_table为空
    target_table_data = sql_obj.select_data("SELECT * from %s" % (target_table))
    if target_table_data:
        print('target_table 非空')
        raise NameError
    # 去除是外卖方式的单子
    select_order = "SELECT * from %s ORDER BY code ASC" % (source_table)
    source_datas = sql_obj.select_data(select_order)
    #print(source_datas)
    pass
    # 程序插入太慢 1000条73s 利用先生成.sql文件再手动导入
    # for a_data in source_datas:
    #     print(a_data)
    #     insert_order = "INSERT INTO {} VALUES {};".format(target_table, a_data)
    #     sql_obj.insert(insert_order)
    with open(target_table + ".sql", "w", encoding='utf-8') as f:
        for a_data in source_datas:
            insert_order = "INSERT INTO {} VALUES {};\n".format(target_table, a_data)
            f.write(insert_order)


def generate_code_items_table(sql_obj,table,save_dir,code_index,item_index,people_num_index): # 生成订单物品对应表
    all_data = sql_obj.get_all_datas(table)
    all_data_list = []
    for i in all_data:  # 数据从元组转化为列表
        all_data_list.append(list(i))
    all_data_list = pd.DataFrame(all_data_list)
    all_codes = pd.value_counts(all_data_list.iloc[:, code_index]).index.values
    all_items = pd.value_counts(all_data_list.iloc[:, item_index]).index.values
    all_data_list = all_data_list.values
    t1 = time.time()
    code_items = dict()  # 订单-物品字典
    for code in all_codes:
        code_items[code] = [0]  # 统计每种菜的个数 可以有重复的
    for d in tqdm(all_data_list):
        # 加入订单有几个人（存在一个单号对应不同的用餐人数）
        if float(d[people_num_index]) > float(code_items[d[code_index]][0]):
            code_items[d[code_index]][0] = d[people_num_index]
        # 将订单的菜加入
        code_items[d[code_index]].append(d[item_index])
    print("整理菜单用时:", time.time())
    print(code_items)
    for k, v in tqdm(code_items.items()):
        code_items_table = pd.DataFrame(np.array(list(v)).reshape((1, -1)), index=[k], columns=None)
        # print(code_items_table)
        code_items_table.to_csv(save_dir, index=True, header=False ,mode='a')


def people_average_num_food(csv_dir, save_dir):
    code_items_csv = pd.read_csv(csv_dir, low_memory=False, index_col=0, header=None, dtype=str).fillna('0')
    people_num_kind = list(pd.value_counts(code_items_csv[1]).index)  # header是None 第一列索引是1不是0
    people_num_kind = sorted(people_num_kind)
    # print(people_num_kind)
    different_peoples_dict = OrderedDict()
    # 根据人数不同将各个订单分开
    for i in people_num_kind:
        different_peoples_dict[i] = code_items_csv[code_items_csv[1] == i]
    #print(different_peoples_dict)
    different_people_average_item = OrderedDict()  # 每种人数的订单平均每种类型的菜买几个
    for k, v in different_peoples_dict.items():
        v = v.values
        each_item_num = np.zeros(10)
        for i in v:  # i是一条log
            for j in i:  # j是一个物品
                if len(j) == 6:
                    each_item_num[int(j[0:2])-1] += 1
                if len(j) == 5:
                    each_item_num[int(j[0]) - 1] += 1
        #print(np.shape(v))
        different_people_average_item[k] = each_item_num/np.shape(v)[0]
    save_table = np.concatenate([v.reshape((1, -1)) for k, v in different_people_average_item.items()], axis=0)
    save_table = pd.DataFrame(save_table, index=people_num_kind, columns=list(range(1, 11)))
    save_table.to_csv(save_dir)


def statistics_item_price(table, save_dir, item_index, price_index):  # 生成物品价格表
    all_data = sql_obj.get_all_datas(table)
    all_data_list = []
    for i in tqdm(all_data):  # 数据从元组转化为列表
        all_data_list.append(list(i))
    all_data_list = pd.DataFrame(all_data_list)
    all_items = pd.value_counts(all_data_list.iloc[:, item_index]).index.values
    all_data_list = all_data_list.values
    #print(all_items)
    items_price_profit = OrderedDict()
    for i in all_items:
        items_price_profit[i] = np.array([0, 0], dtype=float)
    for a_log in all_data_list:
        price = float(a_log[price_index])
        if price > items_price_profit[a_log[item_index]][0]:
            items_price_profit[a_log[item_index]][0] = price
            items_price_profit[a_log[item_index]][1] = round(price*random.uniform(0.4, 0.55),2)
    save_table = np.concatenate([v.reshape((1, -1)) for k, v in items_price_profit.items()], axis=0)
    save_table = pd.DataFrame(save_table, index=all_items, columns=['price','profit'])
    save_table.to_csv(save_dir,)


def generate_items_items_table(sql_obj, table, code_index, item_index):  # 生成两两物品同时出现次数表
    global item_item_matrix
    # code_index 为订单号的索引 32维数据中为4
    t1 = time.time()
    all_data = sql_obj.get_all_datas(table)
    all_data_list = []
    for i in tqdm(all_data):  # 数据从元组转化为列表
        all_data_list.append(list(i))
    print("加载数据用时：", time.time()-t1)

    all_data_list = pd.DataFrame(all_data_list)
    all_codes = pd.value_counts(all_data_list.iloc[:, code_index]).index.values
    all_items = pd.value_counts(all_data_list.iloc[:, item_index]).index.values
    all_data_list = all_data_list.values

    # 整理一个订单对应对个菜
    t1 = time.time()
    code_items = dict()  # 订单-物品字典
    for code in all_codes:
        code_items[code] = set()  # 用set 生成相似度表一单里不能有重复的
    for d in tqdm(all_data_list):
        code_items[d[code_index]].add(d[item_index])
    print("整理菜单用时:", time.time())

    item_item_matrix = pd.DataFrame(np.zeros((len(all_items), len(all_items))),
                                    index=all_items, columns=all_items)
    t1 = time.time()
    item_item_matrix = single_process_get_item_matrix(item_item_matrix, code_items)
    print("生成关系表用时:", time.time()-t1)
    if os.path.exists('./item_item_matrix.csv'):
        os.remove('./item_item_matrix.csv')
    item_item_matrix.to_csv('./item_item_matrix.csv')

def single_process_get_item_matrix(item_item_matrix, code_items):
    for key, value in tqdm(code_items.items()):
        if len(value) == 1:
            continue
        value = list(value)
        for i in range(len(value)-1):
            for j in range(i+1, len(value)):
                item_item_matrix.loc[value[i], value[j]] += 1
                item_item_matrix.loc[value[j], value[i]] += 1
    return item_item_matrix


def mult_process_get_item_matrix(item_item_matrix, code_items):
    def run(value):
        value = list(value)
        for i in range(len(value) - 1):
            for j in range(i + 1, len(value)):
                item_item_matrix.loc[value[i], value[j]] += 1
                item_item_matrix.loc[value[j], value[i]] += 1
        return 1

    executor = ThreadPoolExecutor(max_workers=20)
    print(code_items)
    for _ in executor.map(run, code_items.values()):
        pass
    return item_item_matrix


class Similarity():
    def __init__(self, items_items_matrix_dir, sale_rank_dir):
        self.items_items_matrix = pd.read_csv(items_items_matrix_dir,
                                 header=0, index_col=0)
        #列名读进来自动前边补个0 去掉
        columns = list(self.items_items_matrix.columns)
        for i in range(len(columns)):
            if columns[i][0] == '0':
                columns[i] = columns[i][1:]
        #行名float->str
        index = self.items_items_matrix.index
        index = [str(i) for i in list(index)]
        self.items_items_matrix.columns = columns
        self.items_items_matrix.index = index

        self.sale_rank_table = pd.read_csv(sale_rank_dir,
                                 header=None, index_col=0)
        index = self.sale_rank_table.index
        index = [str(i) for i in index]
        self.sale_rank_table.index = index
        self.sale_rank_table.columns = ['food']
        self.food_list = index
        # print(self.food_list,len(self.food_list))
        # print(self.items_items_matrix)

    def compute_similarity(self, i_name, j_name):
        # 计算ij之间的相似度
        if len(i_name) == 6 and i_name[0] == '0':
            i_name = i_name[1:]
        if len(j_name) == 6 and j_name[0] == '0':
            j_name = j_name[1:]
        a = float(self.items_items_matrix.loc[i_name, j_name])
        b = (float(self.sale_rank_table.loc[i_name, 'food'])*\
             float(self.sale_rank_table.loc[j_name, 'food']))**0.5
        Wij = a/b
        return Wij

    def get_similarity_table(self):
        # 获得相似度表
        table_column = table_index = list(self.items_items_matrix.columns)
        similarity_matrix = np.zeros((len(table_column), len(table_column)))
        similarity_matrix = pd.DataFrame(similarity_matrix,
                                         columns=table_column, index=table_index)
        for i in tqdm(table_index):
            for j in table_column:
                if i == j:
                    similarity_matrix.loc[i, j] = 0
                else:
                    similarity_matrix.loc[i, j] = self.compute_similarity(i, j)
        print(similarity_matrix)
        return similarity_matrix


def generate_similarity_table():  # 生成相关系数表
    r_obj = Similarity('./tables/item_item_matrix.csv', './tables/sale_rank.csv')
    similarity_matrix = r_obj.get_similarity_table()
    similarity_matrix.to_csv('./tables/similarity_matrix.csv')


if __name__ == '__main__':
    # 1000_qinghua_billdetail
    sql_obj = mmySQL('localhost', 'root','123456789', 'dsjxtjc')
    # get_sales_order(sql_obj,'qinghua_billdetail_deletewaimai')
    # get_sales_order_in_diffclass( './tables/sale_rank.csv')
    # delete_waimai(sql_obj,'qinghua_billdetail_origin','qinghua_billdetail_delete_waimai')
    # generate_table_by_order(sql_obj, 'qinghua_billdetail_deletewaimai', 'qinghua_billdetail_byorder')
    # delete_waimai(sql_obj, 'qinghua_billdetail_origin', 'qinghua_billdetail_deletewaimai')
    # generate_items_items_table(sql_obj,'qinghua_billdetail_deletewaimai', 4, 8)
    # generate_similarity_table()
    # generate_code_items_table(sql_obj, 'qinghua_billdetail_deletewaimai', './tables/code_items_table.csv', 4, 8, -11)
    #people_average_num_food('./tables/code_items_table.csv', './tables/people_average_num_food.csv')
    t1 = time.time()
    statistics_item_price('qinghua_billdetail_deletewaimai', './tables/item_price.csv', 8, 16)
    print(time.time()-t1)