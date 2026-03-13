%This is to scale data to make different attributes in the same range.
%Called by stript: dsetpp_test
function descr=data_scale(descr,dimen)

    if ~exist('dimen','var')
        dimen=1;                %in the column direction
    end
    
    if dimen==1
        [n,d]=size(descr);

        for i=1:d
            vec=descr(:,i);
            v_max=max(abs(vec));
            
            if v_max~=0
                descr(:,i)=vec/v_max;
            end
        end
    else
        [d,n]=size(descr);

        for i=1:d
            vec=descr(i,:);
            v_max=max(vec);
            v_min=min(vec);
            den=v_max-v_min;
            
            if den~=0
                descr(i,:)=(vec-v_min)/den;
            end
        end
    end

end