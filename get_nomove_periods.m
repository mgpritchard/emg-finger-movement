close all
clear all
here=pwd;
if not(strcmp(here(end-3:end),'Task'))
    cd('..')
end



split_write_nomoves('victory_gesture',168,72)

main_gestures={'index_finger','middle_finger','ring_finger','little_finger','thumb','rest'};
for gest=1:1:length(main_gestures)
    split_write_nomoves(string(main_gestures(gest)),350,150)
end


function split_write_nomoves(gesturename,trainsize,testsize)
    croppedpath=strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_cropped_and_arranged/');
    train_save_path=strcat(pwd,'/working-dataset/gestures-raw/nomovement_traintest/');
    val_save_path=strcat(pwd,'/working-dataset/gestures-raw/nomovement_validate/');
    spare_save_path=strcat(pwd,'/working-dataset/gestures-raw/nomovement_spare/');

    EMG1=csvread(strcat(croppedpath,gesturename,'/electrode_1.csv'));
    EMG2=csvread(strcat(croppedpath,gesturename,'/electrode_2.csv'));
    EMG3=csvread(strcat(croppedpath,gesturename,'/electrode_3.csv'));
    EMG4=csvread(strcat(croppedpath,gesturename,'/electrode_4.csv'));
    EMG5=csvread(strcat(croppedpath,gesturename,'/electrode_5.csv'));
    EMG6=csvread(strcat(croppedpath,gesturename,'/electrode_6.csv'));
    EMG7=csvread(strcat(croppedpath,gesturename,'/electrode_7.csv'));
    EMG8=csvread(strcat(croppedpath,gesturename,'/electrode_8.csv'));
    
    if strcmp(gesturename,'rest')
        EMGRaw = csvread(strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_unprocessed/',gesturename,'_finger_motion_raw.csv'));
    else
        EMGRaw = csvread(strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_unprocessed/',gesturename,'_motion_raw.csv'));
    end
    
    gestnameshort = split(gesturename,'_');
    for i=1:1:trainsize
        movement_EMG1=EMG1(i,:);
        loc_EMG1=strfind(transpose(EMGRaw(:,1)),movement_EMG1);
        endloc_EMG1=loc_EMG1+150;
        nextloc_EMG1=strfind(transpose(EMGRaw(:,1)),EMG1(i+1,:));
        
       
        nomove_EMG=abs(EMGRaw(endloc_EMG1:nextloc_EMG1,:));
        nomove_tab=array2table(nomove_EMG,...
            'VariableNames',{'EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8'});
        nomove_tab.Timestamp=transpose(1:nextloc_EMG1+1-endloc_EMG1);
        nomove_tab=nomove_tab(:,{'Timestamp','EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8'});
        
        filename='1-'+string(gestnameshort(1))+'-'+string(i)+'.csv';
        writetable(nomove_tab,strcat(train_save_path,filename));
    end
    
    for i=trainsize+1:1:trainsize+testsize
        movement_EMG1=EMG1(i,:);
        loc_EMG1=strfind(transpose(EMGRaw(:,1)),movement_EMG1);
        endloc_EMG1=loc_EMG1+150;
        nextloc_EMG1=strfind(transpose(EMGRaw(:,1)),EMG1(i+1,:));
        
        nomove_EMG=abs(EMGRaw(endloc_EMG1:nextloc_EMG1,:));
        nomove_tab=array2table(nomove_EMG,...
            'VariableNames',{'EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8'});
        nomove_tab.Timestamp=transpose(1:nextloc_EMG1+1-endloc_EMG1);
        nomove_tab=nomove_tab(:,{'Timestamp','EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8'});
        
        filename='1-'+string(gestnameshort(1))+'-'+string(i)+'.csv';
        writetable(nomove_tab,strcat(val_save_path,filename));
    end
    
    for i=trainsize+testsize+1:1:length(EMG1)-1
        movement_EMG1=EMG1(i,:);
        loc_EMG1=strfind(transpose(EMGRaw(:,1)),movement_EMG1);
        endloc_EMG1=loc_EMG1+150;
        nextloc_EMG1=strfind(transpose(EMGRaw(:,1)),EMG1(i+1,:));
        
        nomove_EMG=abs(EMGRaw(endloc_EMG1:nextloc_EMG1,:));
        nomove_tab=array2table(nomove_EMG,...
            'VariableNames',{'EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8'});
        nomove_tab.Timestamp=transpose(1:nextloc_EMG1+1-endloc_EMG1);
        nomove_tab=nomove_tab(:,{'Timestamp','EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8'});
        
        filename='1-'+string(gestnameshort(1))+'-'+string(i)+'.csv';
        writetable(nomove_tab,strcat(spare_save_path,filename));
    end
end
