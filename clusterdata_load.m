%This is to load the data for clustering.
function [descr,label,fname,name_data]=clusterdata_load(idx,direc,flag_tsne)

    if ~exist('flag_tsne','var')
        flag_tsne=0;
    end

    if ~exist('direc','var')
        direc='E:\work\dataset\clustering\';
    end
    
    name_data=list_dataset();
    
    if flag_tsne==0
        direc=[direc,'trued\'];
        fname=[direc,name_data{idx},'.txt'];
    else
        direc=[direc,'2d\'];
        fname=[direc,name_data{idx},'-2d-tsne.txt'];
    end
    
    descr=dlmread(fname);
    dimen=size(descr,2);
    label=descr(:,dimen);
    descr=descr(:,1:dimen-1);

end

%This to to list the datasets used for clustering.

function name_cell=list_dataset(idx)

    nset=100;
    name_data=cell(1,nset);
    name_data(1)={'Aggregation'};    %788 * 2 * 7
    name_data(2)={'Compound'};       %399 * 2 * 6
    name_data(3)={'Pathbased'};      %300 * 2 * 3
    name_data(4)={'Spiral'};         %312 * 2 * 3
    name_data(5)={'D31'};            %3100 * 2 * 31, gaussian
    name_data(6)={'R15'};            %600 * 2 * 15, gaussian
    name_data(7)={'Jain'};           %373 * 2 * 2
    name_data(8)={'Flame'};          %240 * 2 * 2
    name_data(9)={'Mouse'};          %500 * 2 * 3, gaussian, noise
    name_data(10)={'Unbalance'};     %6500 * 2 * 8, gaussian, cluster size
    name_data(11)={'Varydensity'};   %150 * 2 * 3, gaussian, density
    name_data(12)={'S1'};            %5000 * 2 * 15, gaussian, overlap
    name_data(13)={'S2'};            %5000 * 2 * 15, gaussian, overlap
    name_data(14)={'A1'};            %3000 * 2 * 20, gaussian, overlap
    name_data(15)={'A2'};            %5250 * 2 * 35, gaussian, overlap
    name_data(16)={'A3'};            %7500 * 2 * 50, gaussian, overlap
    name_data(17)={'G2-2-10'};       %2048 * 2 * 2, gaussian, overlap and dimension
    name_data(18)={'G2-2-30'};       %2048 * 2 * 2, gaussian, overlap and dimension
    name_data(19)={'G2-2-50'};       %2048 * 2 * 2, gaussian, overlap and dimension
    name_data(20)={'G2-2-100'};      %2048 * 2 * 2, gaussian, overlap and dimension
    name_data(21)={'G2-128-30'};     %2048 * 128 * 2, gaussian, overlap and dimension
    name_data(22)={'G2-1024-50'};    %2048 * 1024 * 2, gaussian, overlap and dimension
    name_data(23)={'Dim032'};        %1024 * 32 * 16, gaussian, dimension
    name_data(24)={'Dim064'};        %1024 * 64 * 16, gaussian, dimension
    name_data(25)={'Dim128'};        %1024 * 128 * 16, gaussian, dimension
    name_data(26)={'Dim256'};        %1024 * 256 * 16, gaussian, dimension
    name_data(27)={'Dim512'};        %1024 * 512 * 16, gaussian, dimension
    name_data(28)={'Dim1024'};       %1024 * 1024 * 16, gaussian, dimension
    name_data(29)={'Spread-2-10'};   %1000 * 2 * 10
    name_data(30)={'Spread-10-20'};  %2000 * 10 * 20
    name_data(31)={'Spread-20-35'};  %3500 * 20 * 35
    name_data(32)={'Spread-35-2'};   %200 * 35 * 2
    name_data(33)={'Spread-50-50'};  %5000 * 50 * 50
    
    name_data(51)={'Thyroid'};        %215 * 5 * 3
    name_data(52)={'Wine'};           %178 * 13 * 3
    name_data(53)={'Iris'};           %150 * 4 * 3
    name_data(54)={'Glass'};          %214 * 9 * 6
    name_data(55)={'Wdbc'};           %569 * 30 * 2
    name_data(56)={'Yeast'};          %1484 * 8 * 10
    name_data(57)={'Breast'};         %699 * 9 * 2
    name_data(58)={'Leaves'};         %1600 * 64 * 100
    name_data(59)={'Seeds'};          %210 * 7 * 3
    name_data(60)={'Segment'};        %2310 * 19 * 7
    name_data(61)={'Libras'};         %360 * 90 * 15
    name_data(62)={'Ionosphere'};     %351 * 34 * 2
    name_data(63)={'Waveform'};       %5000 * 21 * 3
    name_data(64)={'Waveform_noise'}; %5000 * 40 * 3
    name_data(65)={'Ecoli'};          %336 * 7 * 8
    name_data(66)={'CNAE9'};          %1080 * 856 * 9
    name_data(67)={'Olivertti'};      %400 * 28 * 40
    name_data(68)={'Dermatology'};    %366 * 33 * 6
    name_data(69)={'Balance-scale'};  %625 * 4 * 3
    name_data(70)={'Appendicitis'};   %106 * 7 * 2
    name_data(71)={'Arcene'};         %200 * 10000 * 2
    name_data(72)={'Optdigits'};      %5620 * 64 * 10
    name_data(73)={'Robotnavi'};      %5456 * 24 * 4
    name_data(74)={'SCC'};            %600 * 60 * 6
    name_data(75)={'Pendigits'};      %10992 * 16 * 10
    name_data(76)={'USPS'};           %11000 * 256 * 10
    name_data(77)={'Rice'};           %3810 * 7 * 2
    name_data(78)={'DryBean'};        %13611 * 16 * 7
    name_data(79)={'Raisin'};         %900 * 7 * 2
    name_data(80)={'Redwine'};        %1599 * 11 * 6
    name_data(81)={'Whitewine'};      %4898 * 11 * 7
    name_data(82)={'Spambase'};       %4601 * 57 * 2
    name_data(83)={'Gamma_telescope'};%19020 * 10 * 2
    name_data(84)={'Sonar'};          %208 * 60 * 2
    name_data(85)={'Banknote'};       %1372 * 4 * 2
    name_data(86)={'Landsat'};        %6435 * 36 * 6
    name_data(87)={'Vehicle'};        %846 * 18 * 4
    name_data(88)={'Isolet'};         %7797 * 617 * 26
    name_data(89)={'Landmine'};       %338 * 3 * 5
    name_data(90)={'Dutchnumeral'};   %2000 * 649 * 10
    name_data(91)={'Indianliver'};    %583 * 10 * 3
    name_data(92)={'CTG'};            %2126 * 21 * 10
    name_data(93)={'Maternal'};       %1014 * 6 * 3
    name_data(94)={'Pageblocks'};     %5473 * 10 * 5
    name_data(95)={'Hayesroth'};      %160 * 4 * 3
    name_data(96)={'Spectf'};         %267 * 44 * 2
    name_data(97)={'Parkinson'};      %756 * 753 * 2
    name_data(98)={'RNASeq'};         %801 * 20531 * 5
    name_data(99)={'Shuttle'};        %58000 * 9 * 7
    name_data(100)={'Mnist'};         %70000 * 784 * 10
    name_data(101)={'led'};           %500 * 7 * 10
    name_data(102)={'letter'};        %20000 * 16 * 26
    name_data(103)={'vote'};          %435 * 16 * 2  
    name_data(104)={'zoo'};           %101 * 16 * 7
    
    if ~exist('idx','var')
        name_cell=name_data;
    elseif length(idx)>1
        name_cell=name_data(idx);
    else
        name_cell=name_data{idx};
    end

end