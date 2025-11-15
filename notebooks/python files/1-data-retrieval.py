#!/usr/bin/env python
# coding: utf-8

# ## Data-preprocessing

# ### Downloading the Youtube transcripts

# In[2]:


import os
import pandas as pd
import yt_dlp
import json
from webvtt import WebVTT


# In[ ]:


#downloads youtube video subtitles only
def download_subtitles_only(playlist_url, max_videos=60, output_dir='transcripts'):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'skip_download': True,  # Do not download video or audio
        'writesubtitles': True,  # Download uploaded subtitles
        'writeautomaticsub': True,  # Fallback to auto-generated if no uploaded subs
        'subtitleslangs': ['en'],  # English only
        'playlistend': max_videos,
        'outtmpl': f'{output_dir}/%(playlist_index)s - %(title)s.%(ext)s',
        'quiet': False,
        'ignoreerrors': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

    # After download, convert all .vtt files in output_dir to JSON and TXT
    for filename in os.listdir(output_dir):
        if filename.endswith(".vtt"):
            vtt_path = os.path.join(output_dir, filename)
            base_name = filename.rsplit('.', 1)[0]

            # Load .vtt captions
            captions = []
            for caption in WebVTT().read(vtt_path):
                captions.append({
                    "start": caption.start,  # string like "00:00:01.000"
                    "end": caption.end,
                    "text": caption.text.strip()
                })

            # Save JSON
            json_path = os.path.join(output_dir, base_name + ".json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(captions, jf, indent=2, ensure_ascii=False)

            # Save plain text (concatenate all captions)
            txt_path = os.path.join(output_dir, base_name + ".txt")
            with open(txt_path, "w", encoding="utf-8") as tf:
                for caption in captions:
                    tf.write(caption["text"] + "\n")

            print(f"Converted {filename} to {json_path} and {txt_path}")

# Example usage
playlist_url = "https://www.youtube.com/playlist?list=PLnaXrumrax3X8_6L1yL3cejSMH9oTpxiI"
download_subtitles_only(playlist_url)


# ### Convert text to a Pandas DataFrame

# In[ ]:


def load_txt_transcripts_to_df(transcript_dir):
    all_rows = []

    for filename in os.listdir(transcript_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(transcript_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Clean lines: strip newline & spaces
            lines = [line.strip() for line in lines if line.strip() != '']

            # Append rows with source filename
            for line in lines:
                all_rows.append({
                    'text': line,
                    'source_file': filename
                })

    df = pd.DataFrame(all_rows)
    return df
#loads transcripts to directory transcripts
load_txt_transcripts_to_df('/Users/test/Desktop/ironhack_labs/YouTube_ChatBot_Final/datasets/transcripts')


# ### Remove immediate repetitions in the txt files

# In[ ]:


# #removes repetitions in the txt file, but only if the repetition immediately follows the original sentiment. 
# his way we do not lose this rephrasing later in a different context, and the context is still balanced

def clean_repeated_lines_df_and_save(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Process per source file
    for filename, group in df.groupby('source_file'):
        lines = group['text'].tolist()

        cleaned_lines = []
        prev_line = None

        for line in lines:
            line_stripped = line.strip()
            if line_stripped != prev_line:
                cleaned_lines.append(line_stripped)
                prev_line = line_stripped
            else:
                # Skip immediate repeated line
                continue

        # Save cleaned lines to new file in output_dir with same filename
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(line + '\n')

        print(f"Cleaned text saved to: {output_path}")


# Paths
transcript_dir = '/Users/test/Desktop/ironhack_labs/YouTube_ChatBot_Final/datasets/transcripts'
output_clean_dir = '/Users/test/Desktop/ironhack_labs/YouTube_ChatBot_Final/datasets/cleaned_transcripts'


# ### Preprocess the txts

# In[ ]:


def load_txt_files_to_dataframe(directory_path):
    all_rows = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Clean and store lines with metadata
            for line in lines:
                line = line.strip()
                if line:  # skip empty lines
                    all_rows.append({
                        "text": line,
                        "source_file": filename
                    })

    df = pd.DataFrame(all_rows)
    return df

# Usage
directory = "/Users/test/Desktop/ironhack_labs/YouTube_ChatBot_Final/datasets/cleaned_transcripts"
dataframe = load_txt_files_to_dataframe(directory)

# Preview the result
print(dataframe.head())


# ### Pickle the dataframe for further use

# In[ ]:


# Assuming `docs` is a list of LangChain Document objects
dataframe.to_pickle("/Users/test/Desktop/ironhack_labs/YouTube_ChatBot_Final/datasets/dataframe.pkl")

