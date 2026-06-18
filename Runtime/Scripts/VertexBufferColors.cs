// SPDX-FileCopyrightText: 2023 Unity Technologies and the glTFast authors
// SPDX-License-Identifier: Apache-2.0

using System;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
using System.Collections.Generic;
#endif
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

namespace GLTFast
{
    using Logging;
    using Schema;

    sealed class VertexBufferColors : IDisposable
    {
        readonly ICodeLogger m_Logger;

        NativeArray<float4> m_Data;
        readonly bool m_Compact;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
        List<AtomicSafetyHandle> m_SafetyHandles;
#endif

        public VertexBufferColors(int vertexCount, ICodeLogger logger, bool compact = false)
        {
            m_Logger = logger;
            m_Compact = compact;
            Profiler.BeginSample("VertexBufferColors.Allocate");
            m_Data = new NativeArray<float4>(vertexCount, VertexBufferGeneratorBase.defaultAllocator);
            Profiler.EndSample();
        }

        public unsafe bool ScheduleVertexColorJob(
            int colorAccessorIndex,
            int offset,
            NativeArray<JobHandle> handles,
            IGltfBuffers buffers
            )
        {
            Profiler.BeginSample("VertexBufferColors.Schedule");
            buffers.GetAccessorAndData(colorAccessorIndex, out var colorAcc, out var data, out var byteStride);
            if (colorAcc.IsSparse)
            {
                m_Logger?.Error(LogCode.SparseAccessor, "color");
            }

            var colorDestination = m_Data.GetSubArray(offset, colorAcc.count);

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            if (offset > 0)
            {
                // On multi-primitive meshes color jobs may only write into the same destination if we set dedicated
                // safety handles for each sub-array following the first one (offset==0).
                var safetyHandle = AtomicSafetyHandle.Create();
                NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref colorDestination, safetyHandle);
                m_SafetyHandles ??= new List<AtomicSafetyHandle>();
                m_SafetyHandles.Add(safetyHandle);
            }
#endif

            var h = GetColors32Job(
                data,
                colorAcc.componentType,
                colorAcc.GetAttributeType(),
                byteStride,
                colorDestination
            );
            if (h.HasValue)
            {
                handles[0] = h.Value;
            }
            else
            {
                Profiler.EndSample();
                return false;
            }
            Profiler.EndSample();
            return true;
        }

        public void AddDescriptors(VertexAttributeDescriptor[] dst, int offset, int stream)
        {
            dst[offset] = new VertexAttributeDescriptor(
                VertexAttribute.Color,
                m_Compact ? VertexAttributeFormat.UNorm8 : VertexAttributeFormat.Float32,
                4,
                stream);
        }

        public void ApplyOnMesh(
            UnityEngine.Mesh msh,
            int stream,
            MeshUpdateFlags flags = MeshGeneratorBase.defaultMeshUpdateFlags
            )
        {
            Profiler.BeginSample("ApplyUVs");
            if (m_Compact)
            {
                var compactData = new NativeArray<Color32>(m_Data.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (var i = 0; i < m_Data.Length; i++)
                {
                    var color = math.saturate(m_Data[i]) * byte.MaxValue;
                    compactData[i] = new Color32(
                        (byte)math.round(color.x), (byte)math.round(color.y),
                        (byte)math.round(color.z), (byte)math.round(color.w));
                }
                msh.SetVertexBufferData(compactData, 0, 0, compactData.Length, stream, flags);
                compactData.Dispose();
                m_Data.Dispose();
            }
            else
            {
                msh.SetVertexBufferData(m_Data, 0, 0, m_Data.Length, stream, flags);
            }
            Profiler.EndSample();
        }

        public void Dispose()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            if (m_SafetyHandles != null)
            {
                foreach (var handle in m_SafetyHandles)
                {
                    AtomicSafetyHandle.CheckDeallocateAndThrow(handle);
                    AtomicSafetyHandle.Release(handle);
                }
                m_SafetyHandles = null;
            }
#endif
            if (m_Data.IsCreated)
            {
                m_Data.Dispose();
            }
        }

        unsafe JobHandle? GetColors32Job(
            void* input,
            GltfComponentType inputType,
            GltfAccessorAttributeType attributeType,
            int inputByteStride,
            NativeArray<float4> output
            )
        {
            Profiler.BeginSample("PrepareColors32");
            JobHandle? jobHandle = null;

            if (attributeType == GltfAccessorAttributeType.VEC3)
            {
                switch (inputType)
                {
                    case GltfComponentType.UnsignedByte:
                    {
                        var job = new Jobs.ConvertColorsRgbUInt8ToRGBAFloatJob
                        {
                            input = (byte*)input,
                            inputByteStride = inputByteStride > 0 ? inputByteStride : 3,
                            result = output
                        };
                        jobHandle = job.Schedule(output.Length, GltfImport.DefaultBatchCount);
                    }
                    break;
                    case GltfComponentType.Float:
                    {
                        var job = new Jobs.ConvertColorsRGBFloatToRGBAFloatJob
                        {
                            input = (byte*)input,
                            inputByteStride = inputByteStride > 0 ? inputByteStride : 12,
                            result = output
                        };
                        jobHandle = job.Schedule(output.Length, GltfImport.DefaultBatchCount);
                    }
                    break;
                    case GltfComponentType.UnsignedShort:
                    {
                        var job = new Jobs.ConvertColorsRgbUInt16ToRGBAFloatJob
                        {
                            input = (ushort*)input,
                            inputByteStride = inputByteStride > 0 ? inputByteStride : 6,
                            result = output
                        };
                        jobHandle = job.Schedule(output.Length, GltfImport.DefaultBatchCount);
                    }
                    break;
                    default:
                        m_Logger?.Error(LogCode.ColorFormatUnsupported, attributeType.ToString());
                        break;
                }
            }
            else if (attributeType == GltfAccessorAttributeType.VEC4)
            {
                switch (inputType)
                {
                    case GltfComponentType.UnsignedByte:
                    {
                        var job = new Jobs.ConvertColorsRgbaUInt8ToRGBAFloatJob
                        {
                            input = (byte*)input,
                            inputByteStride = inputByteStride > 0 ? inputByteStride : 4,
                            result = output
                        };
                        jobHandle = job.Schedule(output.Length, GltfImport.DefaultBatchCount);
                    }
                    break;
                    case GltfComponentType.Float:
                    {
                        if (inputByteStride == 16 || inputByteStride <= 0)
                        {
                            var job = new Jobs.MemCopyJob
                            {
                                bufferSize = output.Length * 16,
                                input = input,
                                result = output.GetUnsafeReadOnlyPtr()
                            };
                            jobHandle = job.Schedule();
                        }
                        else
                        {
                            var job = new Jobs.ConvertColorsRGBAFloatToRGBAFloatJob
                            {
                                input = (byte*)input,
                                inputByteStride = inputByteStride,
                                result = output
                            };
#if UNITY_COLLECTIONS
                                jobHandle = job.ScheduleBatch(output.Length,GltfImport.DefaultBatchCount);
#else
                            jobHandle = job.Schedule(output.Length, GltfImport.DefaultBatchCount);
#endif
                        }
                    }
                    break;
                    case GltfComponentType.UnsignedShort:
                    {
                        var job = new Jobs.ConvertColorsRgbaUInt16ToRGBAFloatJob
                        {
                            input = (ushort*)input,
                            inputByteStride = inputByteStride > 0 ? inputByteStride : 8,
                            result = output
                        };
#if UNITY_COLLECTIONS
                            jobHandle = job.ScheduleBatch(output.Length,GltfImport.DefaultBatchCount);
#else
                        jobHandle = job.Schedule(output.Length, GltfImport.DefaultBatchCount);
#endif
                    }
                    break;
                    default:
                        m_Logger?.Error(LogCode.ColorFormatUnsupported, attributeType.ToString());
                        break;
                }
            }
            else
            {
                m_Logger?.Error(LogCode.TypeUnsupported, "color accessor", inputType.ToString());
            }
            Profiler.EndSample();
            return jobHandle;
        }
    }
}
