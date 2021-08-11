// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Subway/Particles"
{

    Properties
    {
        _ColorLow("Color Slow Speed", Color) = (0, 0, 0.5, 1)
        _ColorHigh("Color High Speed", Color) = (1, 0, 0, 1)
        _HighSpeedValue("High speed Value", Range(0, 50)) = 25
    }

        SubShader
    {
        Pass
        {
            Blend SrcAlpha one

            CGPROGRAM
            #pragma target 5.0

            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

             struct PS_INPUT
            {
                float4 position : SV_POSITION;
                float4 color : COLOR;
            };

    // Particle's data, shared with the compute shader
    StructuredBuffer<float3> Particles;

    // Properties variables
    uniform float4 _ColorLow;
    uniform float4 _ColorHigh;
    uniform float _HighSpeedValue;

    // Vertex shader
    PS_INPUT vert(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
    {
        PS_INPUT o = (PS_INPUT)0;
        o.color = float4(1.0f, 1.0f, 1.0f, 1.0f);
        o.position = UnityObjectToClipPos(float4(Particles[instance_id],1.0f));

        return o;
    }

    // Pixel shader
    float4 frag(PS_INPUT i) : COLOR
    {
        return i.color;
    }

    ENDCG
}
    }

        Fallback Off
}