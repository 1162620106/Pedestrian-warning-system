function rec = VOCreadxml(path)

if length(path)>5&&strcmp(path(1:5),'http:')
    xml=urlread(path)';
else
    f=fopen(path,'r');
    xml=fread(f,'*char')';
    fclose(f);
end
xml=strrep(xml,'<?xml version="1.0" ?>','');
rec=VOCxml2struct(xml);
