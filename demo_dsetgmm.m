% This is the code of the DSet-GMM algorithm proposed in
% Chongwei Huang, Jian Hou*, Huaqiang Yuan. Enhanced Gaussian Mixture Model
% Clustering for Real-World Data. Engineering Applications of Artificial 
% Intelligence, vol. 173, 114493, June 2026.

function demo_dsetgmm()

    fname='thyroid.txt';

    vec_k=2:50;
    ratio=0.3;
    sigma=8;
    
    res_nmi=zeros(1,length(vec_k));
    
    descr=dlmread(fname);
    dimen=size(descr,2);
    label_t=descr(:,dimen);
    descr=descr(:,1:dimen-1);
        
    dimen=size(descr,2);
        
    if dimen>2
        descr=data_scale(descr);
    end
    
    dima0=pdist(descr,'euclidean');
    dima=squareform(dima0);
    d_mean=mean2(dima); 
    sima=exp(-dima/(d_mean*sigma));

    for nn=1:length(vec_k)
        k=vec_k(nn);
        fprintf([num2str(k),' ']);

        %do clustering
        label_c=dset_gmm(descr,dima,sima,k,ratio);
        
        cq=label2cq(label_c,label_t);
        res_nmi(nn)=cq.nmi;
    end
    
    res_nmi

end

function label_c=dset_gmm(descr,dima,sima,k,ratio)

    %identify boundary points
    label_c=BS(k,ratio,dima,descr);

    core_idx=1:size(dima,2);
    rnn=search_rnn(dima,k,core_idx);

    nidx=find(label_c==-1);
    sima(nidx,:)=0;
    sima(:,nidx)=0;

    %obtain initial labels with sequential dset and expansion
    label_c=extract_dset_filter_core(sima,dima,rnn,label_c);

    %obtain final labels with enhanced gmm
    label_c(label_c==-1)=0;
    label_c=GMM_kailugaji(descr, length(unique(label_c))-1, label_c);

end


function label=BS(k,ratio,dima,descr,flag_refine)

    if ~exist('flag_refine','var')
        flag_refine=1;
    end

    ndata=size(dima,1);
    label=zeros(1,ndata);
    [~, sorted_indices] = sort(dima,2);
    knn=sorted_indices(:,2:k+1);

    nn_num=zeros(1,ndata);
    for ii=1:ndata
        nn_num(ii)=sum(knn(:) == ii);
    end 

    maxnn=max(nn_num(:));
    norm_num=nn_num./maxnn; 

    %calculate cd
    cd=zeros(1,ndata);
    for ii=1:ndata
        tt=knn(ii,:);
        cd(ii)=norm(1/k.*sum(descr(tt,:))-descr(ii,:));
        subdima=dima(ii,tt);
        cd(ii)=cd(ii)/mean(subdima(:));
    end 

    density = (1./(1+norm_num)).*cd;
    
    den1=sort(density,'descend');
    beta=den1(round(ndata*ratio));
    
    label(density>beta)=-1;

    %data refinement
    if flag_refine==1
        bidx=find(density>beta);
        cidx=find(density<=beta);

        label(cidx(nn_num(cidx)<=mean(nn_num(bidx))+std(nn_num(bidx))))=-1;
        label(bidx(nn_num(bidx)>=mean(nn_num(cidx))-std(nn_num(cidx))))=0;
    end

end

function label=extract_dset_filter_core(sima,dima,rnn,label)
    toll=1e-4;
    ndata=size(sima,1);
    min_size=3;
    th_size=min_size+1;             %the minimum size of a cluster
            
    %dset initialization
    for i=1:ndata
        sima(i,i)=0;
    end

    x=zeros(ndata,1);
    core_idx=find(label==0);
    t=length(core_idx);
    x(core_idx)=1/t;
    %start clustering
    num_dsets=0;
    
    while 1>0
        if sum(label==0)<5
            break;
        end
       
        %dset extraction
        x=indyn(sima,x,toll);
        idx_dset=find(x>0);

        if length(idx_dset)<th_size
            break;
        end
        
        num_dsets=num_dsets+1;
        label(idx_dset)=num_dsets;

        label=expand_dset(dima,label,num_dsets,rnn);
            
        %post-processing
        idx=label>0;
        sima(idx,:)=0;
        sima(:,idx)=0;

        idx_ndset=find(label==0);
        num_ndset=length(idx_ndset);
        x=zeros(ndata,1);
        x(idx_ndset)=1/num_ndset;

    end

end

function label=expand_dset(dima,label,num_dsets,rnn)

    while 1>0

        idx_ndset=find(label==0);
        idx_dset=label==num_dsets;
        sub_dima=dima(idx_ndset,idx_dset);
        [~,idx_min]=min(sub_dima,[],1);
        idx_out=idx_ndset(idx_min);
        idx_out=unique(idx_out);
        flag=0;
        for ii=1:length(idx_out)
            if label(idx_out(ii))==0
                tt=rnn{idx_out(ii)};
                    if any(label(tt)==num_dsets)
                        label(idx_out(ii))=num_dsets;              
                        flag=1;                       
                    end                       
            end   
        end

        if flag==0
            break;
        end    

    end    

end


function rnn=search_rnn(unvisit_dima,k,unvisit)
    nn_idx=zeros(length(unvisit),k);
    for ii=1:length(unvisit)
        vec=unvisit_dima(ii,:);
        [~,idx]=sort(vec,'ascend');
        nn_idx(ii,:)=idx(2:k+1);
    end  

    rnn=cell(1,length(unvisit));
    for ii=1:length(unvisit)
        rnn{ii}=[];
        for jj=1:length(unvisit)               
            if ~isempty(find(nn_idx(jj,:)==ii, 1))
                rnn{ii}=[rnn{ii},jj];
            end                       
        end
    end 
end