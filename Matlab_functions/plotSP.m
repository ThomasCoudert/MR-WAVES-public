
thickness = 15 ;%mm
FAtrain = ones(1, 100) * 45;
signal = getSliceProfile(FAtrain, thickness);

y = signal(1,:); % Flip Angle for an RF pulse of FA=45°
% y = ones(1, 201) * 45;
x = linspace(-thickness/2, thickness/2, size(signal, 2));
plot(x, y, 'LineWidth', 10);

xlim([-thickness/2, thickness/2]);
ylim([-10, 60]);


ax = gca;
ax.FontSize = 20; 

xlabel('Slice Thickness (mm)', 'FontSize', 20);
ylabel('Flip Angle (\circ)', 'FontSize', 20);
title('Slice Profile for an RF pulse of FA=45\circ', 'FontSize', 30);

