classdef FM
    %FM : factorization machine 
    properties
        w0;
        wi;
        v;
    end
    
    methods
        function this = FM(input_dimansion,latent_dimansion)
            this.w0 = rand;
            this.wi = rand(1,2*input_dimansion);
            this.v = rand(2*input_dimansion,latent_dimansion);
        end
        
        function y_hat = predict(this,X)
            square_part=0;
            for i=1:size(X,1)
                for j=i+1:size(X,1)
                    square_part = square_part+X(i)*X(j)*sum(this.v(i,:).*this.v(j,:)); 
                end
            end
            y_hat = 1/(1+exp(-(this.w0+sum(this.wi .* X)+square_part)));
        end
        
        function loss = Vtrain(this,input,epoch,batch_size,learning_rate)
            
            loss = zeros(1,epoch+1);
            [~,valid_data,~] = dividerand(randperm(size(input,1)),0.8,0.2,0);
            valid_data = input(valid_data,:);
            mes=0;
            for iteration_batch = 1:batch_size
                    selected = valid_data(randperm(batch_size,1),:);
                    X = selected{1};      
                    y = selected{2};
                    
                    y_hat = this.predict(X);
                    mes = mes+(y_hat-y)^2;
            end
                mes = mes/batch_size;               
                fprintf('epoch: %d , loss = %d\n',0,mes);
                loss(1) = mes;
            
            
            for iteration_epoch=1:epoch
                [train_data,valid_data,~] = dividerand(randperm(size(input,1)),0.8,0.2,0);
                train_data = input(train_data,:);
                valid_data = input(valid_data,:);
                
                for iteration_batch = 1:batch_size
                    selected = train_data(randperm(batch_size,1),:);
                    X = selected{1};      
                    y = selected{2};
                    
                    y_hat = this.predict(X);
                    
                    this.w0 = this.w0-learning_rate*1*(y_hat-y);
                    this.wi = this.wi-learning_rate*X*(y_hat-y);                 
                    for i=1:size(this.v,1)
                        for f=1:size(this.v,2)
                            this.v(i,f) = this.v(i,f)-learning_rate*(X(i)*X*this.v(:,f)-this.v(i,f)*X(i)*X(i))*(y_hat-y);
                        end
                    end                   
                end %end batch
                
                mes = 0;
                for iteration_batch = 1:batch_size
                    selected = valid_data(randperm(batch_size,1),:);
                    X = selected{1};      
                    y = selected{2};
                    
                    y_hat = this.predict(X);
                    mes = mes+(y_hat-y)^2;
                end
                mes = mes/batch_size;
                fprintf('epoch: %d , loss = %d\n',iteration_epoch,mes);
%                 if mod(iteration_epoch,10)==0
%                     fprintf('epoch: %d , loss = %d\n',iteration_epoch,mes);
%                 end
                loss(iteration_epoch+1) = mes;
            end %end epoch            
        end %end Vtrain
    end %end methods 
    
end %end class


