import streamlit as st
import os
import tempfile
from pathlib import Path
import time
import video_extractor
from PIL import Image

st.set_page_config(
    page_title="视频前景提取工具",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create necessary folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def main():
    st.title("🎬 视频前景提取工具")
    
    # Sidebar for model selection and parameters
    with st.sidebar:
        st.header("模型选择")
        model_type = st.selectbox(
            "选择模型",
            ["BEN2-Lite", "BiRefNet-Lite"]
        )
        
        st.header("参数设置")
        batch_size = st.slider("批处理大小", 1, 8, 1, 
                              help="增大批处理大小可加快处理速度，但会占用更多GPU内存")
        
        use_webm = st.checkbox("使用WebM格式（带透明通道）", value=True, 
                              help="生成带透明Alpha通道的WebM视频，否则生成MP4")
        
        if not use_webm:
            bg_color = st.color_picker("背景颜色", "#00FF00")
            
        refine_fg = st.checkbox("优化前景", value=False, 
                               help="使用额外的前景优化处理（更慢但效果更好）")
        
        with st.expander("高级选项"):
            custom_fps = st.number_input("输出FPS（0表示使用原视频帧率）", 
                                        value=0, min_value=0, max_value=60)
    
    # Main panel
    col1, col2 = st.columns(2)
    
    # Input section
    with col1:
        st.header("输入")
        uploaded_video = st.file_uploader("上传视频", type=["mp4", "mov", "avi", "webm"])
        
        use_bg_image = st.checkbox("使用背景图片（可选）", value=False)
        uploaded_bg = None
        if use_bg_image:
            uploaded_bg = st.file_uploader("上传背景图片", type=["jpg", "jpeg", "png"])
        
        if uploaded_video is not None:
            # Save the uploaded video to a temporary file
            temp_video_path = os.path.join("uploads", uploaded_video.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            # Display the uploaded video
            st.video(temp_video_path)
            
            # Display background image if uploaded
            if uploaded_bg is not None:
                bg_image = Image.open(uploaded_bg)
                st.image(bg_image, caption="上传的背景图片", use_column_width=True)
                
                # Save the background image
                bg_path = os.path.join("uploads", uploaded_bg.name)
                bg_image.save(bg_path)
            else:
                bg_path = None
            
            # Process button
            if st.button("开始处理", use_container_width=True):
                # Prepare parameters
                if not use_webm and not bg_path:
                    # Convert hex color to RGB tuple
                    color = bg_color.lstrip('#')
                    bg_color_rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                else:
                    bg_color_rgb = (0, 255, 0)  # Default green
                
                with st.spinner("处理中..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total, message=None):
                        progress = min(current / total, 1.0)
                        progress_bar.progress(progress)
                        if message:
                            status_text.text(f"{message}: {current}/{total} 帧 ({progress*100:.1f}%)")
                        else:
                            status_text.text(f"已处理: {current}/{total} 帧 ({progress*100:.1f}%)")
                    
                    # Process video
                    try:
                        output_path = video_extractor.extract_video(
                            video_path=temp_video_path,
                            output_dir="outputs",
                            model_type=model_type,
                            batch_size=batch_size,
                            use_webm=use_webm,
                            bg_path=bg_path,
                            bg_color=bg_color_rgb,
                            refine_foreground=refine_fg,
                            fps=custom_fps,
                            progress_callback=progress_callback
                        )
                        
                        status_text.success("处理完成！")
                        st.session_state.output_video = output_path
                    except Exception as e:
                        st.error(f"处理失败: {str(e)}")
                        st.exception(e)
    
    # Output section
    with col2:
        st.header("输出")
        if 'output_video' in st.session_state and os.path.exists(st.session_state.output_video):
            output_path = st.session_state.output_video
            
            if output_path.endswith('.webm'):
                # For WebM with alpha, display a download button
                st.markdown("WebM视频（带透明通道）已生成。由于浏览器限制，需下载后查看完整效果。")
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        "下载WebM视频（带透明通道）",
                        f,
                        file_name=os.path.basename(output_path),
                        mime="video/webm"
                    )
                    
                # Still try to display it, but transparency will likely not show correctly
                st.video(output_path)
            else:
                # For MP4 videos
                st.video(output_path)
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        "下载处理后的视频",
                        f,
                        file_name=os.path.basename(output_path),
                        mime="video/mp4"
                    )
            
            # Add delete button for cleanup
            if st.button("清除输出"):
                if os.path.exists(output_path):
                    os.remove(output_path)
                st.session_state.pop('output_video', None)
                st.experimental_rerun()
        else:
            st.info("处理后的视频将显示在这里")

if __name__ == "__main__":
    main() 