import timeit 
import os
import csv
from sys import stdout
def binary_search(mylist, find): 
    while len(mylist) > 0: 
        mid = (len(mylist))//2
        if mylist[mid] == find: 
            return True
        elif mylist[mid] < find: 
            mylist = mylist[:mid] 
        else: 
            mylist = mylist[mid + 1:] 
    return False
  
  
def linear_search(mylist, find): 
    for x in mylist: 
        if x == find: 
            return True
    return False
  
  
def binary_time(): 
   
    
    times = timeit.repeat( 
                          repeat = 3, 
                          number = 10000) 
  
    
    print('Binary search time: {}'.format(min(times)))         
  
  
def linear_time(): 
    times = timeit.repeat( 
                          repeat = 3, 
                          number = 10000) 
  
    
    print('Linear search time: {}'.format(min(times)))   
  
if __name__ == "__main__": 
    '''
    linear_time() 
    binary_time() 
    path = os.getcwd() + '\data'
    dir_list = os.listdir(path)
    print(path)
    print(dir_list)
    '''

    # 1. abrir o arquivo
with open('base_conhecimento.csv', encoding='utf-8') as arquivo_referencia:

  # 2. ler a tabela
  tabela = csv.reader(arquivo_referencia, delimiter=',')

  # 3. navegar pela tabela
  for l in tabela:
    id_autor = l[0]
    nome = l[1]

    print(id_autor, nome) # 191149, Diego C B Mariano
