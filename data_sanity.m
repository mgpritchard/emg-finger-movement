close all
clear all
if not(strcmp(here(end-3:end),'Task'))
    cd('..')
end

indexmoveRaw=csvread(strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_unprocessed/index_finger_motion_raw.csv'));
index_EMG1=csvread(strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_cropped_and_arranged/index_finger/electrode_1.csv'));
index_EMG2=csvread(strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_cropped_and_arranged/index_finger/electrode_2.csv'));
index_EMG3=csvread(strcat(pwd,'/electromyography-emg-dataset/Electro-Myography-EMG-Dataset/raw_emg_data_cropped_and_arranged/index_finger/electrode_3.csv'));

first_index_EMG1=index_EMG1(1,:);
first_index_EMG2=index_EMG2(1,:);
first_index_EMG3=index_EMG3(1,:);

first_index_loc_EMG1=strfind(transpose(indexmoveRaw(:,1)),first_index_EMG1);
first_index_loc_EMG2=strfind(transpose(indexmoveRaw(:,2)),first_index_EMG2);
first_index_loc_EMG3=strfind(transpose(indexmoveRaw(:,3)),first_index_EMG3);

second_index_loc_EMG1=strfind(transpose(indexmoveRaw(:,1)),index_EMG1(2,:));
second_index_loc_EMG2=strfind(transpose(indexmoveRaw(:,2)),index_EMG2(2,:));
second_index_loc_EMG3=strfind(transpose(indexmoveRaw(:,3)),index_EMG3(2,:));

twelfth_index_loc_EMG1=strfind(transpose(indexmoveRaw(:,1)),index_EMG1(12,:));
twelfth_index_loc_EMG2=strfind(transpose(indexmoveRaw(:,2)),index_EMG2(12,:));
twelfth_index_loc_EMG3=strfind(transpose(indexmoveRaw(:,3)),index_EMG3(12,:));

fourth_index_loc_EMG1=strfind(transpose(indexmoveRaw(:,1)),index_EMG1(4,:));

figure(1);
subplot(3,1,1),plot(indexmoveRaw(first_index_loc_EMG1:fourth_index_loc_EMG1+200,1));
subplot(3,1,2),plot(indexmoveRaw(first_index_loc_EMG1:fourth_index_loc_EMG1+200,2));
subplot(3,1,3),plot(indexmoveRaw(first_index_loc_EMG1:fourth_index_loc_EMG1+200,3));

figure(2);
subplot(2,1,1),plot(indexmoveRaw(first_index_loc_EMG1:first_index_loc_EMG1+150,2)),ylim([-0.5,0.5]);
subplot(2,1,2),plot(indexmoveRaw(first_index_loc_EMG1+150:second_index_loc_EMG1,2)),ylim([-0.5,0.5]);