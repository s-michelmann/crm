function plot_ecog(plot_vec, mesh_pth,epos,limits, transparency, view_pos, elec_size, plot_mesh, lrb, clrIn,bar_on, nolight)
% quick script to plot electrodes in MNI on a MESH
% the color of the electrode corresponds to the strength of the effect
% @author: Sebastian Michelmann
% @contact: michelmann.seb@gmail.com
% @dependencies: Fieldtrip toolbox (functions: ft_plot_mesh, data: template
% mesh left and right, i.e. surface_pial_left, surface_pial_right)
%
% Input:
% @plot_vec an nx1 vector with the effect to be plotted (e.g. t-values)
%
% @mesh_pth the path to a folder that contains the mesh for left and right
% surface (here: fieldtrip/template/anatomy) the name of the files are: 
% surface_pial_right.mat and surface_pial_left.mat
%
% @epos an nx3 vector of the electrode positions in MNI space
%
% @limits the limits of the colorbar (e.g. max and min of effect)
%
% @transparency the alpha value for the mesh to make it transparent
%
% @view the view of the 3d figure (e.g. [-90 0])
%example:
%
%
% plot_ecog(effect, ...
%     '/User/tools/fieldtrip/template/anatomy/',...
%     epos, [- 0.003], 0.3, [-70 3], 30);

% fieldtrip/template/anatomy/
ld = load(fullfile(mesh_pth, 'surface_pial_right.mat'));
mesh_r = ld.mesh;
ld = load(fullfile(mesh_pth, 'surface_pial_left.mat'));
mesh_l = ld.mesh;
if nargin <11
    bar_on = 0;
end
if nargin <9
    lrb = 2;
elseif lrb == 0
    sel = epos(:,1)<0;
    epos = epos(sel,:);
    plot_vec = plot_vec(sel);
elseif lrb == 1
    sel = epos(:,1)>=0;
    epos = epos(sel,:);
    plot_vec = plot_vec(sel);
else
    lrb = 2;
end
if nargin <8
    plot_mesh = 1;;
end
if nargin <7
    elec_size = 20;
end
if nargin<6
    view_pos = [-90 0];
end
if nargin <5;
    transparency = 0.5; %transparency of the mesh
end
if nargin < 4 || isempty(limits);
    limits = [-max(abs(plot_vec)) max(abs(plot_vec))]; % the colorbar limit
end
if plot_mesh;
    if lrb == 0
        ft_plot_mesh(mesh_l, 'facealpha', transparency);
        hold on;    
        %lighting gouraud;
%         camlight headlight;
    elseif lrb == 1
       
        ft_plot_mesh(mesh_r, 'facealpha', transparency);
        hold on;
%         lighting gouraud;
%         camlight;
    else
        ft_plot_mesh(mesh_l, 'facealpha', transparency);
        hold on;
        ft_plot_mesh(mesh_r, 'facealpha', transparency);
%         lighting gouraud;
%         camlight;
    end
end

% the effect to plot (a vector of double)
%plot_vec = cell2mat(model_eeg.effect);
if isempty(plot_vec);
    % make it white
    set(gcf, 'color', 'w');    
    % change the view to something
    view(view_pos);
    return; 
end
% values on the colorbar @TODO make colorbar an argument
if nargin < 10 || isempty(clrIn)
    try
        %vals = jet(numel(model_eeg.label));
        %vals = (multigradient('preset', 'div.km.BuRd', 'length', size(epos,1)));
        vals = (multigradient([0 0 1; 1 1 1 ; 1 0 0], 'length',256));
    catch
        vals = jet(256);
    end
else
    vals = clrIn;
end

% a bar from min to max with linear spacing
bar_ = linspace(limits(1), limits(2),256);

% find the corresponding colors
clrs = vals(arrayfun(@(x) nearest(bar_, x), plot_vec)',:);

hold on
%plot the electrode positions as dots in the correct color
ft_plot_mesh(epos, 'vertexcolor', clrs,...
    'vertexsize', elec_size);

% add the colorbar and set it right
if bar_on
    colorbar;
else;
end
colormap(vals);
caxis(limits)

% make it white
set(gcf, 'color', 'w');

% change the view to something
view(view_pos);

% add headlight at the end
if nargin <12 || isempty(nolight) || ~nolight
    camlight HEADLIGHT
end