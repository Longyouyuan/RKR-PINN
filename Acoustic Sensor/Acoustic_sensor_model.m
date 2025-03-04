function [dx,y] = Microphone(t, x, u, k1, k2, k3, k4, k5, k6, varargin)
%MICROPHONE Summary of this function goes here
%   Detailed explanation goes here
y = [x(1); x(2)];
dx = [x(3); ...
      x(4); ...
      k1*x(3) + k2*x(1) + 0.5*k3*k4*(x(2)^2) + k4*u; ...
      k5*x(4) - (x(1)-1)*k3*k6*x(2) + k6*1];
end

