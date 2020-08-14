clear 
clc
a=imread('A:\ct.jpg');
b=imread('A:\me.jpg');
a=imresize(a,[256,256]);
b=imresize(b,[256,256]);
a=rgb2gray(a);
b=rgb2gray(b);
subplot(1,3,1),imshow(a),title('INPUT IMAGE CT');
subplot(1,3,2),imshow(b),title('INPUT IMAGE MRI');
[LL1,LH1,HL1,HH1]=dwt2(a,'haar');
[LL2,LH2,HL2,HH2]=dwt2(b,'haar');
b1= imguidedfilter(LL1);
b2= imguidedfilter(LL2);
[r,c]=size(LL1);
LL4=zeros(128,128);
LH4=zeros(128,128);
HL4=zeros(128,128);
HH4=zeros(128,128);q
for i=1:3:(126)
    for j=1:3:(126)
     for k=i:1:(i+3)
         for l=j:1:(j+3)
             d1(k,l)=LL1(k,l);
             d2(k,l)=LL2(k,l);
         end
     end
     //laplacian
     j1=entropy(d1);
     k1=entropy(d2);
      if j1<k1
        for e=i:1:(i+3)
            for s=j:1:(j+3)
                LL4(e,s)=d1(e,s);
            end
        end
       else
          for e=i:1:(i+3)
            for s=j:1:(j+3)
                LL4(e,s)=d2(e,s);
            end
          end
      end
    end
end
for i=1:3:(126)
   for j=1:3:(126)
     for k=i:1:(i+3)
         for l=j:1:(j+3)
             d3(k,l)=LH1(k,l);
             d4(k,l)=LH2(k,1);
             d5(k,l)=HL1(k,l);
             d6(k,l)=HL2(k,1);
             d7(k,l)=HH1(k,l);
             d8(k,l)=HH2(k,1);
         end
     end
     if wiener2(d3,[i+3,j+3])>wiener2(d4,[i+3,j+3])
          for e=i:1:(i+3)
            for s=j:1:(j+3)
                LH4(e,s)=d4(e,s);
            end
          end
      else
         for e=i:1:(i+3)
              for s=j:1:(j+3)
                  LH4(e,s)=d3(e,s);
               end
          end
     end
     if wiener2(d5,[i+3,j+3])>wiener2(d6,[i+3,j+3])
          for e=i:1:(i+3)
            for s=j:1:(j+3)
                HL4(e,s)=d6(e,s);
            end
          end
      else
         for e=i:1:(i+3)
              for s=j:1:(j+3)
                  HL4(e,s)=d5(e,s);
               end
          end
     end
     if wiener2(d7,[i+3,j+3])>wiener2(d8,[i+3,j+3])
          for e=i:1:(i+3)
            for s=j:1:(j+3)
                HH4(e,s)=d8(e,s);
            end     
          end
     else  
         for e=i:1:(i+3)
              for s=j:1:(j+3)
                  HH4(e,s)=d7(e,s);
               end
         end
       end
            
   end
end

yy=idwt2(LL4,LH4,HL4,HH4,'haar');
subplot(1,3,3),imshow(yy,[]);