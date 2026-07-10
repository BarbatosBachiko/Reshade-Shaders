/*--------------------------------------|
| ::     bb Blue Noise               :: |
|---------------------------------------|
| Shared TexBlueNoise / sTexBlueNoise.  |
'--------------------------------------*/

#pragma once

texture TexBlueNoise < source = "SS_BN3.png"; >
{
    Width = 1024;
    Height = 1024;
    Format = RGBA8;
};
sampler sTexBlueNoise
{
    Texture = TexBlueNoise;
    AddressU = Repeat;
    AddressV = Repeat;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
};
