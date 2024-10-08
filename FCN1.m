clear all


%(10x10) 입력 데이터(10bit one-hot encoding)
X1 = [ 
0 0 0 0 0 0 0 0 0 1; %0
1 0 0 0 0 0 0 0 0 0; %1
0 1 0 0 0 0 0 0 0 0; %2
0 0 1 0 0 0 0 0 0 0; %3
0 0 0 1 0 0 0 0 0 0; %4
0 0 0 0 1 0 0 0 0 0; %5
0 0 0 0 0 1 0 0 0 0; %6
0 0 0 0 0 0 1 0 0 0; %7
0 0 0 0 0 0 0 1 0 0; %8
0 0 0 0 0 0 0 0 1 0; %9
];

% (10 * 4) 목표 데이터(4비트 이진수)
D1 = [ 0 0 0 0; %0
    0 0 0 1;    %1
    0 0 1 0;    %2
    0 0 1 1;    %3
    0 1 0 0;    %4
    0 1 0 1;    %5
    0 1 1 0;    %6
    0 1 1 1;    %7
    1 0 0 0;    %8
    1 0 0 1;    %9
];

% 가중치 행렬 초기화
W1 = 2*rand(20, 10) - 1;     % (20x10)
W2 = 2*rand(30, 20) - 1;     % (30x20)
W3 = 2*rand(40, 30) - 1;     % (40x30)
W4 = 2*rand(4, 40) - 1;      % (4x40)

for epoch = 1:10000          %훈련 10000번 반복
    epoch
 % 훈련 과정
    [W1, W2, W3, W4] = FCN1F(W1, W2, W3, W4, X1, D1);
end
save('FCN1 result.mat')

%load('FCN1 result.mat')

N = 10;                 % 데이터 수 
acc=0;
for k = 1:N
x = X1(k, :)';          % (10x1)

v1 = W1*x;              % (20x10)*(10X1)->(20X1)
y1 = Sigmoid(v1);       % (20X1)

v2 = W2*y1;             % (30x20)*(20X1)->(30X1)
y2 = Sigmoid(v2);       % (30X1)

v3 = W3*y2;             % (40x30)*(30X1)->(40X1)
y3 = Sigmoid(v3);       % (40X1)

v = W4*y3;              % (4x40)*(40X1)->(4X1)
y = round(Sigmoid(v));  % (4X1)

y=y';                   %(1X4)

%정확도 계산
if(y==D1(k,:))
    acc=acc+1;
end

%출력
fprintf(1, '%d\t%d\t%d\t%d\n\n', y);

end

%정확도 출력
acc = acc / N;
fprintf('정확도 %.2f%%\n', acc*100);



