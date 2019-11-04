
nclass=max(fae(:,1));
fdatabase.label=fae(:,1);
tr_idx=[]; ts_idx=[];
for jj = 1:nclass,
        idx_label = find(fdatabase.label == jj);
        num = length(idx_label);
        idx_rand = randperm(num);
        tr_num=train_per_image;
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
end
    tr_fea=fae(tr_idx,:);
    ts_fea=fae(ts_idx,:);

    