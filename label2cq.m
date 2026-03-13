
%This function is used to evaluate the clustering quality, by comparison
%with the ground truth.

function cq=label2cq(label_c,label_t)

    min_labelc=min(label_c);
    if min_labelc<1
        label_c=label_c+1-min_labelc;
    end
    min_labelt=min(label_t);
    if min_labelt<1
        label_t=label_t+1-min_labelt;
    end

    label_c=reshape(label_c,size(label_t));
    
%     cq.acc=cluster_acc(label_t,label_c);
    cq.acc = ClusteringAccuracy(label_t,label_c);
    cq.ami=GetAMI(label_t,label_c);
    cq.ari=AdjustRandIndex(label_t,label_c);
    cq.nmi=nmi_adopt(label_t,label_c);
    cq.rand=label2rand(label_t,label_c);
    cq.vmeasure=label2vmeasure(label_t,label_c);
    cq.fmeasure=label2fmeasure(label_t,label_c);

end

%%%%%%%%%%%%%%%%%%%%%%% accuracy
function acc=cluster_acc(label,pred)
%Find the clustering accuracy of prediction, given the true labels
%Output
% acc = Accuracy of clustering results

%Input
% label = a vector of true labels
% pred = a vector of the predicted labels

% Written by Dong Dong (dongdongwork@gmail.com).

    ytrue=int64(label);
    ypred=int64(pred);

    n=length(ytrue); % number of samples
    m=length(ypred); % number of samples in the prediction
    if n~=m 
        error('The dimensions of two vectors do not match');
    end
    s=min([ytrue(:);ypred(:)])-1; 
    if s<0 %make sure all labels are positive
        ytrue=ytrue-s;
        ypred=ypred-s;
    end
    D=max([ytrue(:);ypred(:)]); % get the largest label
    w=zeros(D);
    for i=1:n  %get the confusion matrix
        w(ypred(i),ytrue(i))=w(ypred(i),ytrue(i))+1;
    end
    M=matchpairs(w, -1, 'max'); %solve the linear assignment problem
    acc=sum(w(sub2ind(size(w), M(:,1), M(:,2))))/n;

end




function Acc = ClusteringAccuracy(label,rlabel)
    % %  Clustering Accuracy
    % % Deng Cai, Xiaofei He, and Jiawei Han,
    %"Document Clustering Using Locality Preserving Indexing", in IEEE TKDE, 2005. 

    % two subcode
    % bestMap, hungarian

    % Input
    % label    real label
    % rlabel   cluster label

    % Output
    % Acc      Accuracy
    
    res = bestMap(label,rlabel)';
    res=reshape(res,size(label));
    Acc = length(find(label == res))/length(label);
%     Acc = sum(label == res)/length(label);

    clear res;
end

function [newL2] = bestMap(L1,L2)
    %bestmap: permute labels of L2 to match L1 as good as possible
    %   [newL2] = bestMap(L1,L2);
    %
    %   version 2.0 --May/2007
    %   version 1.0 --November/2003
    %
    %   Written by Deng Cai (dengcai AT gmail.com)


    %===========    

    L1 = L1(:);
    L2 = L2(:);
    if size(L1) ~= size(L2)
        error('size(L1) must == size(L2)');
    end

    Label1 = unique(L1);
    nClass1 = length(Label1);
    Label2 = unique(L2);
    nClass2 = length(Label2);

    nClass = max(nClass1,nClass2);
    G = zeros(nClass);
    for i=1:nClass1
        for j=1:nClass2
            G(i,j) = length(find(L1 == Label1(i) & L2 == Label2(j)));
        end
    end

    [c,t] = hungarian(-G);
    newL2 = zeros(size(L2));
    for i=1:nClass2
        if c(i) > nClass1
            continue;
        end
        newL2(L2 == Label2(i)) = Label1(c(i));
    end

    clear G;
end

function [C,T]=hungarian(A)
    %HUNGARIAN Solve the Assignment problem using the Hungarian method.
    %
    %[C,T]=hungarian(A)
    %A - a square cost matrix.
    %C - the optimal assignment.
    %T - the cost of the optimal assignment.
    %s.t. T = trace(A(C,:)) is minimized over all possible assignments.

    % Adapted from the FORTRAN IV code in Carpaneto and Toth, "Algorithm 548:
    % Solution of the assignment problem [H]", ACM Transactions on
    % Mathematical Software, 6(1):104-111, 1980.

    % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
    %                 Department of Computing Science, Ume?University,
    %                 Sweden. 
    %                 All standard disclaimers apply.

    % A substantial effort was put into this code. If you use it for a
    % publication or otherwise, please include an acknowledgement or at least
    % notify me by email. /Niclas

    [m,n]=size(A);

    if (m~=n)
        error('HUNGARIAN: Cost matrix must be square!');
    end

    % Save original cost matrix.
    orig=A;

    % Reduce matrix.
    A=hminired(A);

    % Do an initial assignment.
    [A,C,U]=hminiass(A);

    % Repeat while we have unassigned rows.
    while (U(n+1))
        % Start with no path, no unchecked zeros, and no unexplored rows.
        LR=zeros(1,n);
        LC=zeros(1,n);
        CH=zeros(1,n);
        RH=[zeros(1,n) -1];

        % No labelled columns.
        SLC=[];

        % Start path in first unassigned row.
        r=U(n+1);
        % Mark row with end-of-path label.
        LR(r)=-1;
        % Insert row first in labelled row set.
        SLR=r;

        % Repeat until we manage to find an assignable zero.
        while (1)
            % If there are free zeros in row r
            if (A(r,n+1)~=0)
                % ...get column of first free zero.
                l=-A(r,n+1);

                % If there are more free zeros in row r and row r in not
                % yet marked as unexplored..
                if (A(r,l)~=0 & RH(r)==0)
                    % Insert row r first in unexplored list.
                    RH(r)=RH(n+1);
                    RH(n+1)=r;

                    % Mark in which column the next unexplored zero in this row
                    % is.
                    CH(r)=-A(r,l);
                end
            else
                % If all rows are explored..
                if (RH(n+1)<=0)
                    % Reduce matrix.
                    [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
                end

                % Re-start with first unexplored row.
                r=RH(n+1);
                % Get column of next free zero in row r.
                l=CH(r);
                % Advance "column of next free zero".
                CH(r)=-A(r,l);
                % If this zero is last in the list..
                if (A(r,l)==0)
                    % ...remove row r from unexplored list.
                    RH(n+1)=RH(r);
                    RH(r)=0;
                end
            end

            % While the column l is labelled, i.e. in path.
            while (LC(l)~=0)
                % If row r is explored..
                if (RH(r)==0)
                    % If all rows are explored..
                    if (RH(n+1)<=0)
                        % Reduce cost matrix.
                        [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);
                    end

                    % Re-start with first unexplored row.
                    r=RH(n+1);
                end

                % Get column of next free zero in row r.
                l=CH(r);

                % Advance "column of next free zero".
                CH(r)=-A(r,l);

                % If this zero is last in list..
                if(A(r,l)==0)
                    % ...remove row r from unexplored list.
                    RH(n+1)=RH(r);
                    RH(r)=0;
                end
            end

            % If the column found is unassigned..
            if (C(l)==0)
                % Flip all zeros along the path in LR,LC.
                [A,C,U]=hmflip(A,C,LC,LR,U,l,r);
                % ...and exit to continue with next unassigned row.
                break;
            else
                % ...else add zero to path.

                % Label column l with row r.
                LC(l)=r;

                % Add l to the set of labelled columns.
                SLC=[SLC l];

                % Continue with the row assigned to column l.
                r=C(l);

                % Label row r with column l.
                LR(r)=l;

                % Add r to the set of labelled rows.
                SLR=[SLR r];
            end
        end
    end

    % Calculate the total cost.
    T=sum(orig(logical(sparse(C,1:size(orig,2),1))));

end

function A=hminired(A)
    %HMINIRED Initial reduction of cost matrix for the Hungarian method.
    %
    %B=assredin(A)
    %A - the unreduced cost matris.
    %B - the reduced cost matrix with linked zeros in each row.

    % v1.0  96-06-13. Niclas Borlin, niclas@cs.umu.se.

    [m,n]=size(A);

    % Subtract column-minimum values from each column.
    colMin=min(A);
    A=A-colMin(ones(n,1),:);

    % Subtract row-minimum values from each row.
    rowMin=min(A')';
    A=A-rowMin(:,ones(1,n));

    % Get positions of all zeros.
    [i,j]=find(A==0);

    % Extend A to give room for row zero list header column.
    A(1,n+1)=0;
    for k=1:n
        % Get all column in this row. 
        cols=j(k==i)';
        % Insert pointers in matrix.
        A(k,[n+1 cols])=[-cols 0];
    end
end

function [A,C,U]=hminiass(A)
    %HMINIASS Initial assignment of the Hungarian method.
    %
    %[B,C,U]=hminiass(A)
    %A - the reduced cost matrix.
    %B - the reduced cost matrix, with assigned zeros removed from lists.
    %C - a vector. C(J)=I means row I is assigned to column J,
    %              i.e. there is an assigned zero in position I,J.
    %U - a vector with a linked list of unassigned rows.

    % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    [n,np1]=size(A);

    % Initalize return vectors.
    C=zeros(1,n);
    U=zeros(1,n+1);

    % Initialize last/next zero "pointers".
    LZ=zeros(1,n);
    NZ=zeros(1,n);

    for i=1:n
        % Set j to first unassigned zero in row i.
        lj=n+1;
        j=-A(i,lj);

        % Repeat until we have no more zeros (j==0) or we find a zero
        % in an unassigned column (c(j)==0).

        while (C(j)~=0)
            % Advance lj and j in zero list.
            lj=j;
            j=-A(i,lj);

            % Stop if we hit end of list.
            if (j==0)
                break;
            end
        end

        if (j~=0)
            % We found a zero in an unassigned column.

            % Assign row i to column j.
            C(j)=i;

            % Remove A(i,j) from unassigned zero list.
            A(i,lj)=A(i,j);

            % Update next/last unassigned zero pointers.
            NZ(i)=-A(i,j);
            LZ(i)=lj;

            % Indicate A(i,j) is an assigned zero.
            A(i,j)=0;
        else
            % We found no zero in an unassigned column.

            % Check all zeros in this row.

            lj=n+1;
            j=-A(i,lj);

            % Check all zeros in this row for a suitable zero in another row.
            while (j~=0)
                % Check the in the row assigned to this column.
                r=C(j);

                % Pick up last/next pointers.
                lm=LZ(r);
                m=NZ(r);

                % Check all unchecked zeros in free list of this row.
                while (m~=0)
                    % Stop if we find an unassigned column.
                    if (C(m)==0)
                        break;
                    end

                    % Advance one step in list.
                    lm=m;
                    m=-A(r,lm);
                end

                if (m==0)
                    % We failed on row r. Continue with next zero on row i.
                    lj=j;
                    j=-A(i,lj);
                else
                    % We found a zero in an unassigned column.

                    % Replace zero at (r,m) in unassigned list with zero at (r,j)
                    A(r,lm)=-j;
                    A(r,j)=A(r,m);

                    % Update last/next pointers in row r.
                    NZ(r)=-A(r,m);
                    LZ(r)=j;

                    % Mark A(r,m) as an assigned zero in the matrix . . .
                    A(r,m)=0;

                    % ...and in the assignment vector.
                    C(m)=r;

                    % Remove A(i,j) from unassigned list.
                    A(i,lj)=A(i,j);

                    % Update last/next pointers in row r.
                    NZ(i)=-A(i,j);
                    LZ(i)=lj;

                    % Mark A(r,m) as an assigned zero in the matrix . . .
                    A(i,j)=0;

                    % ...and in the assignment vector.
                    C(j)=i;

                    % Stop search.
                    break;
                end
            end
        end
    end

    % Create vector with list of unassigned rows.

    % Mark all rows have assignment.
    r=zeros(1,n);
    rows=C(C~=0);
    r(rows)=rows;
    empty=find(r==0);

    % Create vector with linked list of unassigned rows.
    U=zeros(1,n+1);
    U([n+1 empty])=[empty 0];
end

function [A,C,U]=hmflip(A,C,LC,LR,U,l,r)
    %HMFLIP Flip assignment state of all zeros along a path.
    %
    %[A,C,U]=hmflip(A,C,LC,LR,U,l,r)
    %Input:
    %A   - the cost matrix.
    %C   - the assignment vector.
    %LC  - the column label vector.
    %LR  - the row label vector.
    %U   - the 
    %r,l - position of last zero in path.
    %Output:
    %A   - updated cost matrix.
    %C   - updated assignment vector.
    %U   - updated unassigned row list vector.

    % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    n=size(A,1);

    while (1)
        % Move assignment in column l to row r.
        C(l)=r;

        % Find zero to be removed from zero list..

        % Find zero before this.
        m=find(A(r,:)==-l);

        % Link past this zero.
        A(r,m)=A(r,l);

        A(r,l)=0;

        % If this was the first zero of the path..
        if (LR(r)<0)
            ...remove row from unassigned row list and return.
            U(n+1)=U(r);
            U(r)=0;
            return;
        else

            % Move back in this row along the path and get column of next zero.
            l=LR(r);

            % Insert zero at (r,l) first in zero list.
            A(r,l)=A(r,n+1);
            A(r,n+1)=-l;

            % Continue back along the column to get row of next zero in path.
            r=LC(l);
        end
    end

end

function [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
    %HMREDUCE Reduce parts of cost matrix in the Hungerian method.
    %
    %[A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
    %Input:
    %A   - Cost matrix.
    %CH  - vector of column of 'next zeros' in each row.
    %RH  - vector with list of unexplored rows.
    %LC  - column labels.
    %RC  - row labels.
    %SLC - set of column labels.
    %SLR - set of row labels.
    %
    %Output:
    %A   - Reduced cost matrix.
    %CH  - Updated vector of 'next zeros' in each row.
    %RH  - Updated vector of unexplored rows.

    % v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    n=size(A,1);

    % Find which rows are covered, i.e. unlabelled.
    coveredRows=LR==0;

    % Find which columns are covered, i.e. labelled.
    coveredCols=LC~=0;

    r=find(~coveredRows);
    c=find(~coveredCols);

    % Get minimum of uncovered elements.
    m=min(min(A(r,c)));

    % Subtract minimum from all uncovered elements.
    A(r,c)=A(r,c)-m;

    % Check all uncovered columns..
    for j=c
        % ...and uncovered rows in path order..
        for i=SLR
            % If this is a (new) zero..
            if (A(i,j)==0)
                % If the row is not in unexplored list..
                if (RH(i)==0)
                    % ...insert it first in unexplored list.
                    RH(i)=RH(n+1);
                    RH(n+1)=i;
                    % Mark this zero as "next free" in this row.
                    CH(i)=j;
                end
                % Find last unassigned zero on row I.
                row=A(i,:);
                colsInList=-row(row<0);
                if (length(colsInList)==0)
                    % No zeros in the list.
                    l=n+1;
                else
                    l=colsInList(row(colsInList)==0);
                end
                % Append this zero to end of list.
                A(i,l)=-j;
            end
        end
    end

    % Add minimum to all doubly covered elements.
    r=find(coveredRows);
    c=find(coveredCols);

    % Take care of the zeros we will remove.
    [i,j]=find(A(r,c)<=0);

    i=r(i);
    j=c(j);

    for k=1:length(i)
        % Find zero before this in this row.
        lj=find(A(i(k),:)==-j(k));
        % Link past it.
        A(i(k),lj)=A(i(k),j(k));
        % Mark it as assigned.
        A(i(k),j(k))=0;
    end

    A(r,c)=A(r,c)+m;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ami
%Program for calculating the Adjusted Mutual Information (AMI) between
%two clusterings, tested on Matlab 7.0 (R14)
%(C) Nguyen Xuan Vinh 2008-2010
%Contact: n.x.vinh@unsw.edu.au 
%         vthesniper@yahoo.com
%--------------------------------------------------------------------------
%**Input: a contingency table T
%   OR
%        cluster label of the two clusterings in two vectors
%        eg: true_mem=[1 2 4 1 3 5]
%                 mem=[2 1 3 1 4 5]
%        Cluster labels are coded using positive integer. 
%**Output: AMI: adjusted mutual information  (AMI_max)
%
%**Note: In a prevous published version, if you observed strange AMI results, eg. AMI>>1, 
%then it's likely that in these cases the expected MI was incorrectly calculated (the EMI is the sum
%of many tiny elements, each falling out the precision range of the computer).
%However, you'll likely see that in those cases, the upper bound for the EMI will be very
%tiny, and hence the AMI -> NMI (see [3]). It is recommended setting AMI=NMI in
%these cases, which is implemented in this version.
%--------------------------------------------------------------------------
%References: 
% [1] 'A Novel Approach for Automatic Number of Clusters Detection based on Consensus Clustering', 
%       N.X. Vinh, and Epps, J., in Procs. IEEE Int. Conf. on 
%       Bioinformatics and Bioengineering (Taipei, Taiwan), 2009.
% [2] 'Information Theoretic Measures for Clusterings Comparison: Is a
%	    Correction for Chance Necessary?', N.X. Vinh, Epps, J. and Bailey, J.,
%	    in Procs. the 26th International Conference on Machine Learning (ICML'09)
% [3] 'Information Theoretic Measures for Clusterings Comparison: Variants, Properties, 
%       Normalization and Correction for Chance', N.X. Vinh, Epps, J. and
%       Bailey, J., Journal of Machine Learning Research, 11(Oct), pages
%       2837-2854, 2010

function [AMI_]=GetAMI(true_mem,mem)
    if nargin==1
        T=true_mem; %contingency table pre-supplied
    elseif nargin==2
        %build the contingency table from membership arrays
        R=max(true_mem);
        C=max(mem);
        n=length(mem);N=n;

        %identify & removing the missing labels
        list_t=ismember(1:R,true_mem);
        list_m=ismember(1:C,mem);
        T=Contingency(true_mem,mem);
        T=T(list_t,list_m);
    end

    %-----------------------calculate Rand index and others----------
    n=sum(sum(T));N=n;
    C=T;
    nis=sum(sum(C,2).^2);		%sum of squares of sums of rows
    njs=sum(sum(C,1).^2);		%sum of squares of sums of columns

    t1=nchoosek(n,2);		%total number of pairs of entities
    t2=sum(sum(C.^2));      %sum over rows & columnns of nij^2
    t3=.5*(nis+njs);

    %Expected index (for adjustment)
    nc=(n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1));

    A=t1+t2-t3;		%no. agreements
    D=  -t2+t3;		%no. disagreements

    if t1==nc
       AR=0;			%avoid division by zero; if k=1, define Rand = 0
    else
       AR=(A-nc)/(t1-nc);		%adjusted Rand - Hubert & Arabie 1985
    end

    RI=A/t1;			%Rand 1971		%Probability of agreement
    MIRKIN=D/t1;	    %Mirkin 1970	%p(disagreement)
    HI=(A-D)/t1;      	%Hubert 1977	%p(agree)-p(disagree)
    Dri=1-RI;           %distance version of the RI
    Dari=1-AR;          %distance version of the ARI
    %-----------------------%calculate Rand index and others%----------

    %update the true dimensions
    [R C]=size(T);
    if C>1 a=sum(T');else a=T';end;
    if R>1 b=sum(T);else b=T;end;

    %calculating the Entropies
    Ha=-(a/n)*log(a/n)'; 
    Hb=-(b/n)*log(b/n)';

    %calculate the MI (unadjusted)
    MI=0;
    for i=1:R
        for j=1:C
            if T(i,j)>0 MI=MI+T(i,j)*log(T(i,j)*n/(a(i)*b(j)));end;
        end
    end
    MI=MI/n;

    %-------------correcting for agreement by chance---------------------------
    AB=a'*b;
    bound=zeros(R,C);
    sumPnij=0;

    E3=(AB/n^2).*log(AB/n^2);

    EPLNP=zeros(R,C);
    LogNij=log([1:min(max(a),max(b))]/N);
    for i=1:R
        for j=1:C
            sumPnij=0;
            nij=max(1,a(i)+b(j)-N);
            X=sort([nij N-a(i)-b(j)+nij]);
            if N-b(j)>X(2)
                nom=[[a(i)-nij+1:a(i)] [b(j)-nij+1:b(j)] [X(2)+1:N-b(j)]];
                dem=[[N-a(i)+1:N] [1:X(1)]];
            else
                nom=[[a(i)-nij+1:a(i)] [b(j)-nij+1:b(j)]];       
                dem=[[N-a(i)+1:N] [N-b(j)+1:X(2)] [1:X(1)]];
            end
            p0=prod(nom./dem)/N;

            sumPnij=p0;

            EPLNP(i,j)=nij*LogNij(nij)*p0;
            p1=p0*(a(i)-nij)*(b(j)-nij)/(nij+1)/(N-a(i)-b(j)+nij+1);  

            for nij=max(1,a(i)+b(j)-N)+1:1:min(a(i), b(j))
                sumPnij=sumPnij+p1;
                EPLNP(i,j)=EPLNP(i,j)+nij*LogNij(nij)*p1;
                p1=p1*(a(i)-nij)*(b(j)-nij)/(nij+1)/(N-a(i)-b(j)+nij+1);            

            end
             CC=N*(a(i)-1)*(b(j)-1)/a(i)/b(j)/(N-1)+N/a(i)/b(j);
             bound(i,j)=a(i)*b(j)/N^2*log(CC);         
        end
    end

    EMI_bound=sum(sum(bound));
    EMI_bound_2=log(R*C/N+(N-R)*(N-C)/(N*(N-1)));
    EMI=sum(sum(EPLNP-E3));

    AMI_=(MI-EMI)/(max(Ha,Hb)-EMI);
    NMI=MI/sqrt(Ha*Hb);


    %If expected mutual information negligible, use NMI.
    if abs(EMI)>EMI_bound
    %     fprintf('The EMI is small: EMI < %f, setting AMI=NMI',EMI_bound);
        AMI_=NMI;
    end

    clear bound T EPLNP;
end

%---------------------auxiliary functions---------------------
function Cont=Contingency(Mem1,Mem2)

    if nargin < 2 || min(size(Mem1)) > 1 || min(size(Mem2)) > 1
       error('Contingency: Requires two vector arguments')
       return
    end

    Cont=zeros(max(Mem1),max(Mem2));

    for i = 1:length(Mem1);
       Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ari
function ARI = AdjustRandIndex(RealLabel,PredictLabel)
    % 本程序编写于2016年12月8日
    % 实现 Adjust Rand Index 
    % 参考维基百科
    % 或 NoteExpress 中的笔记

    % RealLabel 实际标签
    % PredictLabel 预测标签

    % ARI 返回 Adjust Rand Index 值

    n = length(RealLabel);      % 数据集大小
    seq = 1:n;
    Rlabel = unique(RealLabel);
    KR = numel(Rlabel);      % 实际簇个数
    Plabel = unique(PredictLabel);
    KP = numel(Plabel);   % 预测簇个数
    % contingency table
    CT = zeros(KR,KP);
    for i=1:KR
        flagR = RealLabel == Rlabel(i);
        numR = seq(flagR);     % 属于簇 i 的样本编号/序号
        for j=1:KP
            flagP = PredictLabel == Plabel(j);
            numP = seq(flagP);   % 属于簇 j 的样本编号/序号
            % n_{ij} = | X_i \bigcap Y_i |
            CT(i,j) = numel(intersect(numR,numP));
        end
    end
    a = sum(CT,2);
    b = sum(CT,1);
    % 计算组合数
    aR = zeros(KR,1);
    for i=1:KR
        aR(i) = a(i)*(a(i)-1)/2.0;
    end
    bP = zeros(KP,1);
    for i=1:KP
        bP(i) = b(i)*(b(i)-1)/2.0;
    end
    nRP = zeros(KR,KP);
    for i=1:KR
        for j=1:KP
            nRP(i,j) = CT(i,j)*(CT(i,j)-1)/2.0;
        end
    end
    saR = sum(aR);
    sbP = sum(bP);
    snRP = sum(sum(nRP));
    nk2 = n*(n-1)/2.0;
    ARI = (snRP-saR*sbP/nk2)/((saR+sbP)/2.0-(saR*sbP)/nk2);

    clear CT nPR;
end

function [ AR ] = ARI1(Clustering1,k1,Clustering2,k2)
% This function returns Adjusted Rand Index ( Hubert & Arabie) of two clusterings 1 & 2.
%variable 'Clustering1' is Nx1 vector with an integer number between 1
%and K1 to denote which cluster the corresponding data point assigned to in
%the first clustering. Similarly for 'Clustering2'
N=size(Clustering1,1);
contig_matrix= zeros(k1,k2);

for point=1:N
   i=Clustering1(point) ;
   j=Clustering2(point);
  contig_matrix(i, j) = contig_matrix(i, j)+1;
    
end


a= sum(contig_matrix');
b=sum(contig_matrix);

SumCombnij=0;

for i=1:k1
    for j=1:k2
        
    if (contig_matrix(i,j)>1) 
        SumCombnij=SumCombnij+ nchoosek(contig_matrix(i,j),2) ;
    end  
        
        
    end 
end

SumCombai=0;
for i=1:k1
   if ( a(i)>1)
         SumCombai=  SumCombai+nchoosek(a(i),2);
   end 
    
end
SumCombbj=0;
for j=1:k2
   if ( b(j)>1)
         SumCombbj=  SumCombbj+nchoosek(b(j),2);
   end 
    
end
nCh2=nchoosek(N,2);
temp=(SumCombai*SumCombbj)/nCh2;

AR =(SumCombnij-temp)/(0.5*(SumCombai+SumCombbj)-temp);

clear contig_matrix;

end

function ari = adjust_rand_index(Y, Y0)

    Y=reshape(Y, size(Y0,1),size(Y0,2));

    K=max(Y); 
    K0=max(Y0);
    nk=zeros(K,K0);
    for i=1:K
        for j=1:K0
            nk(i,j)=sum((Y==i)&(Y0==j));
        end
    end
    sums=0;
    for i=1:K
        for j=1:K0
            sums=sums+nk(i,j)*(nk(i,j)-1)/2;
        end
    end
    nk1=sum(nk,1);
    sumd1=0;
    for j=1:K0
        sumd1=sumd1+nk1(j)*(nk1(j)-1)/2;
    end
    nk2=sum(nk,2);
    sumd2=0;
    for i=1:K
        sumd2=sumd2+nk2(i)*(nk2(i)-1)/2;
    end
    N=numel(Y); sumN=N*(N-1)/2;
    ari=(sums-sumd1*sumd2/sumN)/(0.5*(sumd1+sumd2)-sumd1*sumd2/sumN);

    clear nk;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% nmi
function z = nmi_adopt(x, y)
    % Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
    % Input:
    %   x, y: two integer vector of the same length 
    % Ouput:
    %   z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
    % Written by Mo Chen (sth4nth@gmail.com).
    assert(numel(x) == numel(y));
    n = numel(x);
    x = reshape(x,1,n);
    y = reshape(y,1,n);

    l = min(min(x),min(y));
    x = x-l+1;
    y = y-l+1;
    k = max(max(x),max(y));

    idx = 1:n;
    Mx = sparse(idx,x,1,n,k,n);
    My = sparse(idx,y,1,n,k,n);
    Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
    Hxy = -dot(Pxy,log2(Pxy));


    % hacking, to elimative the 0log0 issue
    Px = nonzeros(mean(Mx,1));
    Py = nonzeros(mean(My,1));

    % entropy of Py and Px
    Hx = -dot(Px,log2(Px));
    Hy = -dot(Py,log2(Py));

    % mutual information
    MI = Hx + Hy - Hxy;

    % normalized mutual information
    z = sqrt((MI/Hx)*(MI/Hy));
    z = max(0,z);

    clear Hx Hy MI;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% rand
function ri=label2rand(label_c,label_t)

    ndata=length(label_c);
    
    ncount=0;
    ntotal=0;
    for i=1:ndata-1
        for j=i+1:ndata
            if label_c(i)==label_c(j) && label_t(i)==label_t(j)
                ncount=ncount+1;
            end
            if label_c(i)~=label_c(j) && label_t(i)~=label_t(j)
                ncount=ncount+1;
            end
            ntotal=ntotal+1;
        end
    end

    ri=ncount/ntotal;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% F-measure
%This is used to compute the F-measure of two labels, used to
%evaluate the accuracy of clustering.

function score=label2fmeasure(label_c,label_t)

    vlabel_c=unique(label_c);
    nlabel_c=length(vlabel_c);
    
    vlabel_t=unique(label_t);
    nlabel_t=length(vlabel_t);
    
    ndata=length(label_t);

    score=0;
    for i=1:nlabel_t
        lt=vlabel_t(i);
        nlt=length(find(label_t==lt));
        
        sf=zeros(1,nlabel_c);
        
        for j=1:nlabel_c
            lc=vlabel_c(j);
            nlc=length(find(label_c==lc));
            
            num=0;
            for k=1:ndata
                if label_t(k)==lt && label_c(k)==lc
                    num=num+1;
                end
            end
            
            sp=num/nlc;
            sr=num/nlt;
            sf(j)=2*sp*sr/(sp+sr);
        end
        
        score=score+nlt*max(sf);
        clear sf;
    end

    score=score/ndata;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% V-measure
%This is used to calculte V-measure, used to evaluate clustering results.

function [score v_homo v_complete]=label2vmeasure(label_k,label_c)

    vc=unique(label_c);
    nc=length(vc);
    
    vk=unique(label_k);
    nk=length(vk);
    
    ndata=length(label_k);
    
    %contingency table
    ma=zeros(nc,nk);
    for c=1:nc
        lc=vc(c);
        
        for k=1:nk
            lk=vk(k);
            
            for i=1:ndata
                if label_c(i)==lc
                    if label_k(i)==lk
                        ma(c,k)=ma(c,k)+1;
                    end
                end
            end
        end
    end
    
    %homogeneity
    hck=0;
    for k=1:nk
        idx=find(label_k==vk(k));
        nak=length(idx);
        
        sum=0;
        for c=1:nc
            if ma(c,k)>0
                sum=sum+ma(c,k)/ndata*log(ma(c,k)/nak);
            end
        end
        
        hck=hck+sum;
    end
    
    hc=0;
    for c=1:nc
        idx=find(label_c==vc(c));
        nac=length(idx);
        
        hc=hc+nac/ndata*log(nac/ndata);
    end
    
    if hck==0
        v_homo=1;
    else
        v_homo=1-hck/hc;
    end
    
    %completeness
    hkc=0;
    for c=1:nc
        idx=find(label_c==vc(c));
        nac=length(idx);
        
        sum=0;
        for k=1:nk
            if ma(c,k)>0
                sum=sum+ma(c,k)/ndata*log(ma(c,k)/nac);
            end
        end
        
        hkc=hkc+sum;
    end
    
    hk=0;
    for k=1:nk
        idx=find(label_k==vk(k));
        nak=length(idx);
        
        hk=hk+nak/ndata*log(nak/ndata);
    end

    if hkc==0
        v_complete=1;
    else
        v_complete=1-hkc/hk;
    end
    
    %V-measure
    beta=1;
    
    if v_complete==0 && v_homo==0
        score=0;
    else
        score=(1+beta)*v_homo*v_complete/(beta*v_homo+v_complete);
    end

    clear ma; 
end