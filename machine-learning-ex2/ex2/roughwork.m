h = [2 5 7 1];
y = [1 0 1 0];
m = length(y);

h = h';
y = y';

s = 0;
for i = 1:size(h)
	s = s + (y(i) * log(h(i))) + ((1 - y(i)) * log(1 - h(i)));
endfor;
J = -s / m;
