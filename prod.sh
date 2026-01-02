#!/bin/bash
if [ -f "leaderboard.csv" ]; then
    echo "Existing leaderboard.csv found."
    read -p "Do you want to delete it and start fresh? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
	cp leaderboard.csv ./backups/$(date).csv
        rm -f leaderboard.csv
        echo "leaderboard.csv deleted"
        echo
    else
        echo "Keeping existing leaderboard.csv"
        echo
    fi
else
    echo "No existing leaderboard.csv found. Starting fresh."
    echo
fi

echo "Starting Gunicorn server on port 2026..."
echo
gunicorn -w 4 -b 0.0.0.0:2026 --timeout 30 --access-logfile - --error-logfile - app:app
