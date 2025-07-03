#!/bin/bash

BACKUP_DIR="./data_backup"

case "$1" in
    "backup")
        echo "Backing up data..."
        mkdir -p "$BACKUP_DIR"
        docker-compose down
        
        # Backup volumes
        docker run --rm -v echat_sqlite_data:/source -v $(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/sqlite_backup.tar.gz -C /source .
        docker run --rm -v echat_chroma_data:/source -v $(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/chroma_backup.tar.gz -C /source .
        
        echo "Backup completed to $BACKUP_DIR"
        ;;
        
    "restore")
        echo "Restoring data..."
        if [ ! -d "$BACKUP_DIR" ]; then
            echo "No backup directory found!"
            exit 1
        fi
        
        docker-compose down
        
        # Restore volumes
        docker run --rm -v echat_sqlite_data:/dest -v $(pwd)/$BACKUP_DIR:/backup alpine sh -c "cd /dest && tar xzf /backup/sqlite_backup.tar.gz"
        docker run --rm -v echat_chroma_data:/dest -v $(pwd)/$BACKUP_DIR:/backup alpine sh -c "cd /dest && tar xzf /backup/chroma_backup.tar.gz"
        
        echo "Restore completed"
        ;;
        
    *)
        echo "Usage: $0 {backup|restore}"
        exit 1
esac
