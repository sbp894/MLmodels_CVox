clear;
clc;

in_dir= ['level65_dBspl_clean' filesep];
out_dir= ['level65_dBspl_clean_ds' filesep];

all_files= dir([in_dir '**' filesep '*.mat']);
force_redo= 1;

max_wt= 100; %Hz
max_wf= 5;

max_vals= nan(length(all_files),1);

rng(0);

% for fileVar=randsample(1:length(all_files), 20)
parfor fileVar=1:length(all_files)
    cur_fStruct= all_files(fileVar);
    cur_fName_in= [cur_fStruct.folder filesep cur_fStruct.name];
    cur_fName_out= [strrep(cur_fStruct.folder, in_dir, out_dir) filesep 'ds_' cur_fStruct.name];

    if ~exist(cur_fName_out, 'file') || force_redo
        if ~isfolder(fileparts(cur_fName_out))
            mkdir(fileparts(cur_fName_out))
        end

        mps_struct_in=load(cur_fName_in);
        mps_struct= mps_struct_in.mps_struct;

        valid_wt= abs(mps_struct.fft_wt)<max_wt;
        valid_wf= abs(mps_struct.fft_wf)<max_wf;
        mps_struct.fft_wt= mps_struct.fft_wt(valid_wt);
        mps_struct.fft_wf= mps_struct.fft_wf(valid_wf);
        mps_struct.mps_pow= mps_struct.mps_pow(valid_wf, valid_wt);

        if rem(length(mps_struct.fft_wt),3)==1
            % add two cols to make it divisible by 3
            mps_struct.mps_pow= [zeros(size(mps_struct.mps_pow,1),1), mps_struct.mps_pow, zeros(size(mps_struct.mps_pow,1),1)];
            mps_struct.fft_wt= [nan, mps_struct.fft_wt, nan];
        elseif rem(length(mps_struct.fft_wt),3)==2
            % add one col to make it divisible by 3
            mps_struct.mps_pow= [zeros(size(mps_struct.mps_pow,1),1), mps_struct.mps_pow];
            mps_struct.fft_wt= [nan, mps_struct.fft_wt];
        end

        mps_struct.mps_pow_dB= db(( mps_struct.mps_pow(:,1:3:end) + mps_struct.mps_pow(:,2:3:end) + mps_struct.mps_pow(:,3:3:end))); % factor of 20 to scale input b/w 0 and 1
        mps_struct= rmfield(mps_struct, 'mps_pow');
        mps_struct.fft_wt= mps_struct.fft_wt(2:3:end);

        max_vals(fileVar) = max(mps_struct.mps_pow_dB(:));
        parsave(cur_fName_out, mps_struct)

        doDebugPlots= false;
        if doDebugPlots
            figure(2);
            clf;
            subplot(121);
            imagesc(mps_struct.mps_pow_dB)
            title(cur_fStruct.name, 'Interpreter','none')

            subplot(122);
            histogram(mps_struct.mps_pow_dB(:))
            xlim([-200 -60])
        end
    end
end

figure(11);
histogram(max_vals, 50)

function parsave(cur_fName_out, mps_struct)
save(cur_fName_out, 'mps_struct');
end