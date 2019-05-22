clear all
clc

%           S0      S1      S2
Pa=       [ 0.5     0       0.5;  %S0
            0.7     0.1     0.2;  %S1
            0.4     0       0.6]; %S2


%           S0      S1      S2
Pa(:,:,2)=[ 0       0       1;    %S0
            0       0.95    0.05; %S1
            0.3     0.3     0.4]; %S2

%           S0     S1       S2
Ra=       [ 0       0       0;  %S0
            5       0       0;  %S1
            0       0       0]; %S2

%           S0     S1       S2
Ra(:,:,2)=[ 0       0       0;  %S0
            0       0       0;  %S1
            -1      0       0]; %S2

discount = 0.99;
tolerance = 0.0001;
max_iter = 1000;
policy = randi(size(Pa,3),1,size(Pa,2));

 
for i=1:max_iter
    cpu_time = cputime;
    
    %Policy evaluation
    [vector_values] = policy_evaluation(Pa,Ra,policy,discount,tolerance,max_iter);
    
    %Policy update
    [policy_new] = policy_update(Pa,Ra,discount,vector_values);
    
    if policy == policy_new
        fprintf('Value function converged after %i iterations\n',i);
        break
    else
        policy = policy_new      
    end
    cpu_time = cputime - cpu_time
end

function [vector_values, P, R, iterations] = policy_evaluation(Pa,Ra,policy,discount,tolerance,max_iter)
vector_values = zeros(1,size(Pa,2));
old_values = zeros(1,size(Pa,2));
for k=1:length(policy)
    P(k,:)=Pa(k,:,policy(k));
    R(k,:)=Ra(k,:,policy(k));
end
for it=1:max_iter
    for state=1:size(P,2)
        suma=0;
        for snext=1:size(P,2)
            suma = suma + P(state, snext)*(R(state, snext) + discount*old_values(snext));
        end
        vector_values(state) = suma;
    end
    if abs(old_values-vector_values)<tolerance
        iterations = it;
        fprintf('Value function converged after %i iterations\n',it)
        break
    end
    old_values=vector_values;
end
end
function [policy_new] = policy_update(Pa,Ra,discount,vector_values)
policy_new = zeros(1,size(Pa,2));
valor = zeros(1,size(Pa,3));

%Policy update/improvement
    for s=1:size(Pa,2)        
        for a=1:size(Pa,3)
            sum = 0;            
            for sprime=1:size(Pa,2)
                sum = sum + Pa(s,sprime,a)* (Ra(s,sprime,a)+ discount* vector_values(sprime));
            end
            valor(a) = sum;
        end
        [~, policy_new(s)] = max(valor);
    end   
end