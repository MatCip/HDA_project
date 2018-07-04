full_path='column_names.txt'
file = open(full_path,"r")  
dict_={}
keep_cols=np.concatenate([np.arange(2,47),np.arange(51,60),np.arange(64,73),np.arange(77,86),np.arange(90,99),np.arange(103,135)])
print(keep_cols.shape)

indexes=range(0,113)
cont=0;

for line in file:
    
    string=line;
    line_splitted=line.split();
    if(len(line_splitted)!=0):
        
        try:
            numb = int(line_splitted[1])
            if(numb<=243 and numb >1 and np.sum(keep_cols==numb)>0):
                dict_[indexes[cont]]=line_splitted[2]+'_'+line_splitted[3]+'_'+line_splitted[4]+'column_'+line_splitted[1]
                cont=cont+1;
                
        except ValueError:
            print()
            
print(dict_)
output = open('index_to_channel.pkl', 'wb')
pk.dump(dict_, output)
output.close()
    

