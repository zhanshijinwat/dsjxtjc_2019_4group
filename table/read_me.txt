1.code_items_table.csv： 
订单号中包含菜品的表格 第一列是订单号 第二列是这一单里的人数 之后是包含的菜品
读取时候因为每行列数不同 pd.read_csv设置low_memory=False
2.item_item_matrix
两个物品同时出现的单数的表

3.similarity_matrix
两个物品相似度的表

4.sale_rank
各个菜品销量

5.sales_order_in_different_class
每一类菜品的销量

6.people_average_num_food
不同人数订单的菜品构成  第一列是订单用餐人数 三位数的用餐人数应该是登记错的

7.item_price
物品价格及利润

