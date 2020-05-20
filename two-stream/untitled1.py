import os 
file = open("D:/autism/testfile.txt","w") 
file_1 = open("D:/autism/trainfile.txt","w") 

count = 0
for filename in os.listdir('D:/autism/video_combine_S_T'):
    for i in  os.listdir('D:/autism/video_combine_S_T/'+filename): 
        
        if count % 5 == 0:
            file_1.write(filename+"/"+i+"\n")
        else:
            file.write(filename+"/"+i+"\n")
    count+=1
            
        
     
