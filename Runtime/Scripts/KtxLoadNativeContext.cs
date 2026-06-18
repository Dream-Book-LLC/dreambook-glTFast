// SPDX-FileCopyrightText: 2023 Unity Technologies and the glTFast authors
// SPDX-License-Identifier: Apache-2.0

#if KTX_IS_RECENT
#define KTX_IS_ENABLED
#endif

#if KTX_IS_ENABLED

using System.Threading.Tasks;
using KtxUnity;
using Unity.Collections;

namespace GLTFast {
    class KtxLoadNativeContext : KtxLoadContextBase {
        ReadOnlyNativeArray<byte> m_Data;

        public KtxLoadNativeContext(int index,ReadOnlyNativeArray<byte> data) {
            imageIndex = index;
            m_Data = data;
            m_KtxTexture = new KtxTexture();
        }

        public override async Task<TextureResult> LoadTexture2D(bool linear) {
            try {
                var errorCode = m_KtxTexture.Open(m_Data.AsNativeArrayReadOnly());
                if (errorCode != ErrorCode.Success) return new TextureResult(errorCode);
                return await m_KtxTexture.LoadTexture2D(linear);
            }
            finally {
                m_KtxTexture.Dispose();
                m_KtxTexture = null;
                m_Data = default;
            }
        }
    }
}
#endif // KTX_IS_INSTALLED
