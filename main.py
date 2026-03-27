#!/usr/bin/env python
# Parallelized Mastodon Sentiment Analysis using MPI

import json
import time
import argparse
from datetime import datetime
from mpi4py import MPI

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Mastodon Sentiment Analysis using MPI')
    parser.add_argument('file', help='Path to the NDJSON file')
    parser.add_argument('--verbose', '-v', help='Enable verbose output')

    # Only process arguments on rank 0 and broadcast to other processes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        args = parser.parse_args()
        args_dict = vars(args)
    else:
        args_dict = None

    # Broadcast arguments from rank 0 to all processes
    args_dict = comm.bcast(args_dict, root=0)

    # Create an argparse.Namespace object from the dictionary
    args = argparse.Namespace(**args_dict)
    return args

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse arguments
    args = parse_arguments()

    # Start timing the execution
    start_time = time.time()

    # Master process announces the job
    if rank == 0:
        print(f"Starting Mastodon sentiment analysis with {size} processes")
        print(f"Processing file: {args.file}")
        print(f"Process {rank}: I am the master")
    else:
        if args.verbose:
            print(f"Process {rank}: I am a worker")

    file_path = args.file

    # Each process calculates its chunk of the file
    hours_sentiment_data, users_sentiment_data = process_file_chunk(file_path, rank, size, comm, args)

    # Gather all data at the master process
    all_hours_data = comm.gather(hours_sentiment_data, root=0)
    all_users_data = comm.gather(users_sentiment_data, root=0)

    # Master process computes the final results
    if rank == 0:
        # Marge all data
        merged_hours_data = {}
        merged_users_data = {}

        for each_hour in all_hours_data:
            for hour_key, sentiment in each_hour.items():
                if hour_key in merged_hours_data:
                    merged_hours_data[hour_key] += sentiment
                else:
                    merged_hours_data[hour_key] = sentiment

        for each_user in all_users_data:
            for user_key, sentiment in each_user.items():
                if user_key in merged_users_data:
                    merged_users_data[user_key] += sentiment
                else:
                    merged_users_data[user_key] = sentiment

        # Find happiest and saddest hours and users
        happiest_hours = sorted(merged_hours_data.items(), key=lambda x: x[1], reverse=True)[:5]
        saddest_hours = sorted(merged_hours_data.items(), key=lambda x: x[1])[:5]

        happiest_users = sorted(merged_users_data.items(), key=lambda x: x[1], reverse=True)[:5]
        saddest_users = sorted(merged_users_data.items(), key=lambda x: x[1])[:5]

        # print results
        print("5 HAPPIEST HOURS")
        for i, (datetime, sentiment) in enumerate(happiest_hours, 1):
            print(f"({i}) {datetime} : {sentiment}")

        print("=====")

        print("5 SADDEST HOURS")
        for i, (datetime, sentiment) in enumerate(saddest_hours, 1):
            print(f"({i}) {datetime}: {sentiment}")

        print("=====")

        print("5 HAPPIEST USERS")
        for i, (username_and_id, sentiment) in enumerate(happiest_users, 1):
            print(f"({i}) {username_and_id} : {sentiment}")

        print("=====")

        print("5 SADDEST USERS")
        for i, (username_and_id, sentiment) in enumerate(saddest_users, 1):
            print(f"({i}) {username_and_id} : {sentiment}")

        # Report execution time
        end_time = time.time()
        print(f"\nExecution time: {end_time - start_time:} seconds")

def process_file_chunk(file_path, rank, size, comm, args):
    """
    Process a chunk of the file based on the process rank and total size
    """

    # First, determine the file size to calculate chunks
    if rank == 0:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(0)
        except Exception as e:
            print(f"Error reading file: {e}")
            file_size = 0
    else:
        file_size = None

    # Broadcast file size to all processes
    file_size = comm.bcast(file_size, root=0)

    if file_size == 0:
        return {}, {}

    # Calculate chunk size and starting position for each process
    chunk_size = file_size // size
    start_pos = rank * chunk_size
    end_pos = start_pos + chunk_size if rank < size - 1 else file_size

    # Initialize aggregated data dictionaries
    hours_sentiment_data = {}
    users_sentiment_data = {}


    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Seek to the starting position
            f.seek(start_pos)

            # If not the first process, find the beginning of the next complete line
            if rank > 0:
                f.readline()

            current_pos = f.tell()

            while current_pos < end_pos:
                each_line = f.readline()
                if not each_line:
                    break

                current_pos = f.tell()

                if each_line.strip():
                    try:
                        json_obj = json.loads(each_line)  # each json_obj is a dictionary

                        # get doc.createdAt and doc.sentiment
                        each_doc = json_obj.get('doc', {})
                        each_created_at = each_doc.get('createdAt')
                        each_sentiment = float(each_doc.get('sentiment'))

                        if each_sentiment == 0.0:
                            continue

                        # get doc.account.id and doc.account.username
                        each_account = each_doc.get('account', {})
                        each_account_id = each_account.get('id')
                        each_username = each_account.get('username')

                        if each_account_id is None or each_username is None:
                            continue

                        try:
                            dt = datetime.fromisoformat(each_created_at.replace('Z', '+00:00'))

                            # Extract date in the format YYYY-MM-DD and Extract hour in the format HH
                            each_date_part = dt.strftime('%Y-%m-%d')
                            each_hour_part = dt.strftime('%H')
                            am_pm = "PM" if int(each_hour_part) >= 12 else "AM"
                            hour_key = f"{each_date_part} {each_hour_part}-{int(each_hour_part) + 1}{am_pm}"

                            # Aggregate to hour dictionary directly
                            if hour_key in hours_sentiment_data:
                                hours_sentiment_data[hour_key] += each_sentiment
                            else:
                                hours_sentiment_data[hour_key] = each_sentiment

                            # Aggregate to user dictionary directly
                            user_key = f"{each_username} {each_account_id}"
                            if user_key in users_sentiment_data:
                                users_sentiment_data[user_key] += each_sentiment
                            else:
                                users_sentiment_data[user_key] = each_sentiment

                        except (ValueError, TypeError) as e:
                            print(f"Process {rank}: Error parsing date: {e}")
                            continue

                    except json.JSONDecodeError as e:
                        pass

    except Exception as e:
        print(f"Process {rank}: Error processing file chunk: {e}")

    # Only print this if verbose mode is enabled or it's the master process
    if rank == 0 or args.verbose:
        print(f"Process {rank}: Processed {len(hours_sentiment_data)} entries")

    return hours_sentiment_data, users_sentiment_data


if __name__ == "__main__":
    main()

