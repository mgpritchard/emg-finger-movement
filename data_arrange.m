close all
clear all
here=pwd;
if not(strcmp(here(end-3:end),'Task'))
    cd('..')
end


%{
rawpath=strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_unprocessed/');
indexmoveRaw=csvread(strcat(rawpath,'index_finger_motion_raw.csv'));
middlemoveRaw=csvread(strcat(rawpath,'middle_finger_motion_raw.csv'));
ringmoveRaw=csvread(strcat(rawpath,'ring_finger_motion_raw.csv'));
littlemoveRaw=csvread(strcat(rawpath,'little_finger_motion_raw.csv'));
thumbmoveRaw=csvread(strcat(rawpath,'thumb_motion_raw.csv'));
victorymoveRaw=csvread(strcat(rawpath,'victory_gesture_motion_raw.csv'));
restmoveRaw=csvread(strcat(rawpath,'rest_finger_motion_raw.csv'));
%}


croppedpath=strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_cropped_and_arranged/');
index_EMG1=csvread(strcat(croppedpath,'index_finger/electrode_1.csv'));
middle_EMG1=csvread(strcat(croppedpath,'middle_finger/electrode_1.csv'));
ring_EMG1=csvread(strcat(croppedpath,'ring_finger/electrode_1.csv'));
little_EMG1=csvread(strcat(croppedpath,'little_finger/electrode_1.csv'));
thumb_EMG1=csvread(strcat(croppedpath,'thumb/electrode_1.csv'));
victory_EMG1=csvread(strcat(croppedpath,'victory_gesture/electrode_1.csv'));
rest_EMG1=csvread(strcat(croppedpath,'rest/electrode_1.csv'));


train_save_path=strcat(pwd,'/working-dataset/gestures-raw/traintest/');
val_save_path=strcat(pwd,'/working-dataset/gestures-raw/validate/');
spare_save_path=strcat(pwd,'/working-dataset/gestures-raw/validation_spare/');


%{
victory_EMG2=csvread(strcat(croppedpath,'victory_gesture/electrode_2.csv'));
victory_EMG3=csvread(strcat(croppedpath,'victory_gesture/electrode_3.csv'));
victory_EMG4=csvread(strcat(croppedpath,'victory_gesture/electrode_4.csv'));
victory_EMG5=csvread(strcat(croppedpath,'victory_gesture/electrode_5.csv'));
victory_EMG6=csvread(strcat(croppedpath,'victory_gesture/electrode_6.csv'));
victory_EMG7=csvread(strcat(croppedpath,'victory_gesture/electrode_7.csv'));
victory_EMG8=csvread(strcat(croppedpath,'victory_gesture/electrode_8.csv'));

for i=1:1:2
    new=table;
    new.Timestamp=transpose(1:150);
    new.EMG1=abs(transpose(victory_EMG1(i,:)));
    new.EMG2=abs(transpose(victory_EMG2(i,:)));
    new.EMG3=abs(transpose(victory_EMG3(i,:)));
    new.EMG4=abs(transpose(victory_EMG4(i,:)));
    new.EMG5=abs(transpose(victory_EMG5(i,:)));
    new.EMG6=abs(transpose(victory_EMG6(i,:)));
    new.EMG7=abs(transpose(victory_EMG7(i,:)));
    new.EMG8=abs(transpose(victory_EMG8(i,:)));
    filename='1-victory-'+string(i)+'.csv';
    writetable(new,strcat(train_save_path,filename));
end
%}

split_write_tables('victory_gesture',168,72)

main_gestures={'index_finger','middle_finger','ring_finger','little_finger','thumb','rest'};
for gest=1:1:length(main_gestures)
    split_write_tables(string(main_gestures(gest)),350,150)
end



function split_write_tables(gesturename,trainsize,testsize)
    croppedpath=strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_cropped_and_arranged/');    
    train_save_path=strcat(pwd,'/working-dataset/gestures-raw/traintest/');
    val_save_path=strcat(pwd,'/working-dataset/gestures-raw/validate/');
    spare_save_path=strcat(pwd,'/working-dataset/gestures-raw/validation_spare/');

    EMG1=csvread(strcat(croppedpath,gesturename,'/electrode_1.csv'));
    EMG2=csvread(strcat(croppedpath,gesturename,'/electrode_2.csv'));
    EMG3=csvread(strcat(croppedpath,gesturename,'/electrode_3.csv'));
    EMG4=csvread(strcat(croppedpath,gesturename,'/electrode_4.csv'));
    EMG5=csvread(strcat(croppedpath,gesturename,'/electrode_5.csv'));
    EMG6=csvread(strcat(croppedpath,gesturename,'/electrode_6.csv'));
    EMG7=csvread(strcat(croppedpath,gesturename,'/electrode_7.csv'));
    EMG8=csvread(strcat(croppedpath,gesturename,'/electrode_8.csv'));
    
    gestnameshort = split(gesturename,'_');
    for i=1:1:trainsize
        new=table;
        new.Timestamp=transpose(1:150);
        new.EMG1=abs(transpose(EMG1(i,:)));
        new.EMG2=abs(transpose(EMG2(i,:)));
        new.EMG3=abs(transpose(EMG3(i,:)));
        new.EMG4=abs(transpose(EMG4(i,:)));
        new.EMG5=abs(transpose(EMG5(i,:)));
        new.EMG6=abs(transpose(EMG6(i,:)));
        new.EMG7=abs(transpose(EMG7(i,:)));
        new.EMG8=abs(transpose(EMG8(i,:)));
        filename='1-'+string(gestnameshort(1))+'-'+string(i)+'.csv';
        writetable(new,strcat(train_save_path,filename));
    end
    
    for i=trainsize+1:1:trainsize+testsize
        new=table;
        new.Timestamp=transpose(1:150);
        new.EMG1=abs(transpose(EMG1(i,:)));
        new.EMG2=abs(transpose(EMG2(i,:)));
        new.EMG3=abs(transpose(EMG3(i,:)));
        new.EMG4=abs(transpose(EMG4(i,:)));
        new.EMG5=abs(transpose(EMG5(i,:)));
        new.EMG6=abs(transpose(EMG6(i,:)));
        new.EMG7=abs(transpose(EMG7(i,:)));
        new.EMG8=abs(transpose(EMG8(i,:)));
        filename='1-'+string(gestnameshort(1))+'-'+string(i)+'.csv';
        writetable(new,strcat(val_save_path,filename));
    end
    
    for i=trainsize+testsize+1:1:length(EMG1)
        new=table;
        new.Timestamp=transpose(1:150);
        new.EMG1=abs(transpose(EMG1(i,:)));
        new.EMG2=abs(transpose(EMG2(i,:)));
        new.EMG3=abs(transpose(EMG3(i,:)));
        new.EMG4=abs(transpose(EMG4(i,:)));
        new.EMG5=abs(transpose(EMG5(i,:)));
        new.EMG6=abs(transpose(EMG6(i,:)));
        new.EMG7=abs(transpose(EMG7(i,:)));
        new.EMG8=abs(transpose(EMG8(i,:)));
        filename='1-'+string(gestnameshort(1))+'-'+string(i)+'.csv';
        writetable(new,strcat(spare_save_path,filename));
    end
end


